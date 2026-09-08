<?php
/**
 * Optimized BTC Puzzle Server for Real CPU VPS Farms (Speed ~600K - 1.5M keys/s)
 * SQLite WAL Engine - Absolutely no 70,000 file locks generated - No Redis needed
 */

header('Access-Control-Allow-Origin: *');
header('Access-Control-Allow-Methods: GET, POST, OPTIONS');
header('Access-Control-Allow-Headers: Content-Type');

if ($_SERVER['REQUEST_METHOD'] === 'OPTIONS') {
    http_response_code(204);
    exit;
}

const DATA_DIR = __DIR__ . '/data';

function get_results_json_path(int $puzzle_id): string {
    return DATA_DIR . '/results_' . $puzzle_id . '.json';
}

function get_result_txt_path(int $puzzle_id): string {
    return DATA_DIR . '/result_' . $puzzle_id . '.txt';
}

// RANGE SIZE 2^28 KEYS (~268.4M KEYS)
const RANGE_SIZE_EXP       = 28;
const RANGE_SIZE           = '268435456'; // 2^28 keys (~268.4M keys)
const RANGES_PER_BLOCK     = 1024;        // 2^10
const LEASE_TIMEOUT_SECS   = 1200;        // 20 phút timeout (phù hợp range ~3-6 phút)
const MIN_BUFFER_RANGES    = 2000;        // Pre-buffer in SQLite

const DEFAULT_PUZZLE = 71;

const PUZZLES = [
    70 => [
        'puzzle'         => 70,
        'target_address' => '19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR',
        'lower'          => '590295810358705651712', // 2^69
        'upper'          => '1180591620717411303424',
        'total'          => '590295810358705651712',
        'version_byte'   => 0,
        'total_blocks'   => '2147483648', // 2^69 / (2^28 * 1024) = 2^31
        'power_of_2'     => 69,
    ],
    71 => [
        'puzzle'         => 71,
        'target_address' => '1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU',
        'lower'          => '1180591620717411303424', // 2^70
        'upper'          => '2361183241434822606848',
        'total'          => '1180591620717411303424',
        'version_byte'   => 0,
        'total_blocks'   => '4294967296', // 2^70 / (2^28 * 1024) = 2^32
        'power_of_2'     => 70,
    ],
];

function resolve_puzzle_id($raw_id = null): int {
    if ($raw_id === null || $raw_id === '' || !is_numeric($raw_id)) {
        return DEFAULT_PUZZLE;
    }
    $id = (int)$raw_id;
    if (isset(PUZZLES[$id])) {
        return $id;
    }
    return DEFAULT_PUZZLE;
}

if (!is_dir(DATA_DIR)) {
    @mkdir(DATA_DIR, 0777, true);
}

function get_puzzle_db(int $puzzle_id): PDO {
    static $pdos = [];
    if (isset($pdos[$puzzle_id])) return $pdos[$puzzle_id];

    $puzzle_dir = DATA_DIR . '/puzzle_' . $puzzle_id;
    if (!is_dir($puzzle_dir)) {
        @mkdir($puzzle_dir, 0777, true);
    }
    $db_path = $puzzle_dir . '/puzzle_' . $puzzle_id . '.sqlite3';

    $pdo = new PDO('sqlite:' . $db_path, null, null, [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
        PDO::ATTR_TIMEOUT => 5,
    ]);

    $pdo->exec('PRAGMA journal_mode = WAL;');
    $pdo->exec('PRAGMA synchronous = NORMAL;');
    $pdo->exec('PRAGMA busy_timeout = 5000;');
    $pdo->exec('PRAGMA cache_size = -32000;');
    $pdo->exec('PRAGMA auto_vacuum = INCREMENTAL;');

    init_schema($pdo);
    $pdos[$puzzle_id] = $pdo;
    return $pdo;
}

function init_schema(PDO $pdo): void {
    $pdo->exec("
        CREATE TABLE IF NOT EXISTS done_intervals (
            puzzle_id INTEGER NOT NULL,
            start_block INTEGER NOT NULL,
            end_block INTEGER NOT NULL,
            PRIMARY KEY (puzzle_id, start_block)
        ) WITHOUT ROWID;

        CREATE INDEX IF NOT EXISTS idx_intervals_lookup ON done_intervals(puzzle_id, start_block, end_block);

        CREATE TABLE IF NOT EXISTS blocks (
            puzzle_id INTEGER NOT NULL,
            block_id INTEGER NOT NULL,
            status INTEGER DEFAULT 0,
            done_count INTEGER DEFAULT 0,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (puzzle_id, block_id)
        );

        CREATE TABLE IF NOT EXISTS ranges (
            puzzle_id INTEGER NOT NULL,
            block_id INTEGER NOT NULL,
            range_idx INTEGER NOT NULL,
            status INTEGER DEFAULT 0,
            worker TEXT DEFAULT '',
            claimed_at INTEGER DEFAULT 0,
            PRIMARY KEY (puzzle_id, block_id, range_idx)
        );

        CREATE INDEX IF NOT EXISTS idx_ranges_claim ON ranges(puzzle_id, status, claimed_at);
        CREATE INDEX IF NOT EXISTS idx_ranges_block ON ranges(puzzle_id, block_id, status);

        CREATE TABLE IF NOT EXISTS user_stats (
            worker TEXT PRIMARY KEY,
            speed REAL DEFAULT 0,
            ranges_done INTEGER DEFAULT 0,
            current_block INTEGER DEFAULT 0,
            current_range INTEGER DEFAULT 0,
            last_seen INTEGER NOT NULL
        );
    ");

    try {
        @$pdo->exec("ALTER TABLE user_stats ADD COLUMN current_block INTEGER DEFAULT 0;");
    } catch (Exception $e) {}
    try {
        @$pdo->exec("ALTER TABLE user_stats ADD COLUMN current_range INTEGER DEFAULT 0;");
    } catch (Exception $e) {}
}

function respond(array $payload, int $code = 200): void {
    header('Content-Type: application/json; charset=utf-8');
    http_response_code($code);
    echo json_encode($payload, JSON_UNESCAPED_UNICODE);
    exit;
}

function error_resp(string $msg, int $code = 400): void {
    respond(['error' => $msg], $code);
}

function is_block_done_interval(PDO $pdo, int $puzzle_id, int $block_id): bool {
    $stmt = $pdo->prepare("
        SELECT end_block FROM done_intervals 
        WHERE puzzle_id = ? AND start_block <= ? 
        ORDER BY start_block DESC LIMIT 1
    ");
    $stmt->execute([$puzzle_id, $block_id]);
    $row = $stmt->fetch();
    if (!$row) return false;
    return ((int)$row['end_block'] >= $block_id);
}

function mark_block_done_interval(PDO $pdo, int $puzzle_id, int $block_id): void {
    $st_left = $pdo->prepare("SELECT start_block, end_block FROM done_intervals WHERE puzzle_id = ? AND end_block = ?");
    $st_left->execute([$puzzle_id, $block_id - 1]);
    $left = $st_left->fetch();

    $st_right = $pdo->prepare("SELECT start_block, end_block FROM done_intervals WHERE puzzle_id = ? AND start_block = ?");
    $st_right->execute([$puzzle_id, $block_id + 1]);
    $right = $st_right->fetch();

    if ($left && $right) {
        $pdo->prepare("DELETE FROM done_intervals WHERE puzzle_id = ? AND start_block = ?")
            ->execute([$puzzle_id, $right['start_block']]);
        $pdo->prepare("UPDATE done_intervals SET end_block = ? WHERE puzzle_id = ? AND start_block = ?")
            ->execute([$right['end_block'], $puzzle_id, $left['start_block']]);
    } elseif ($left) {
        $pdo->prepare("UPDATE done_intervals SET end_block = ? WHERE puzzle_id = ? AND start_block = ?")
            ->execute([$block_id, $puzzle_id, $left['start_block']]);
    } elseif ($right) {
        $pdo->prepare("DELETE FROM done_intervals WHERE puzzle_id = ? AND start_block = ?")
            ->execute([$puzzle_id, $right['start_block']]);
        $pdo->prepare("INSERT INTO done_intervals (puzzle_id, start_block, end_block) VALUES (?, ?, ?)")
            ->execute([$puzzle_id, $block_id, $right['end_block']]);
    } else {
        $pdo->prepare("INSERT OR IGNORE INTO done_intervals (puzzle_id, start_block, end_block) VALUES (?, ?, ?)")
            ->execute([$puzzle_id, $block_id, $block_id]);
    }
}

function compact_completed_block(PDO $pdo, int $puzzle_id, int $block_id): void {
    mark_block_done_interval($pdo, $puzzle_id, $block_id);
    $pdo->prepare("DELETE FROM ranges WHERE puzzle_id = ? AND block_id = ?")->execute([$puzzle_id, $block_id]);
    $pdo->prepare("DELETE FROM blocks WHERE puzzle_id = ? AND block_id = ?")->execute([$puzzle_id, $block_id]);
    $pdo->exec("PRAGMA incremental_vacuum;");
}

function ensure_range_buffer(PDO $pdo, int $puzzle_id, array $config): void {
    $now = time();
    $active_workers = (int)$pdo->query("SELECT COUNT(*) FROM user_stats WHERE last_seen > " . ($now - 600))->fetchColumn();
    $target_buffer = max(MIN_BUFFER_RANGES, min(50000, $active_workers * 3));

    $stmt = $pdo->prepare("SELECT COUNT(*) FROM ranges WHERE puzzle_id = ? AND status = 0");
    $stmt->execute([$puzzle_id]);
    $todo_count = (int)$stmt->fetchColumn();

    if ($todo_count >= $target_buffer) return;

    $total_blocks = (int)$config['total_blocks'];
    if ($total_blocks <= 0) return;

    $needed_ranges = $target_buffer - $todo_count;
    $blocks_needed = min(10, max(1, (int)ceil($needed_ranges / RANGES_PER_BLOCK)));

    $pdo->beginTransaction();
    try {
        $ins_blk = $pdo->prepare("INSERT OR IGNORE INTO blocks (puzzle_id, block_id, status, done_count, created_at) VALUES (?, ?, 1, 0, ?)");
        $ins_rng = $pdo->prepare("INSERT OR IGNORE INTO ranges (puzzle_id, block_id, range_idx, status, worker, claimed_at) VALUES (?, ?, ?, 0, '', 0)");
        $chk_active = $pdo->prepare("SELECT 1 FROM blocks WHERE puzzle_id = ? AND block_id = ? LIMIT 1");

        for ($b = 0; $b < $blocks_needed; $b++) {
            $chosen_block = null;
            for ($att = 0; $att < 30; $att++) {
                $cand = random_int(0, $total_blocks - 1);
                if (is_block_done_interval($pdo, $puzzle_id, $cand)) continue;

                $chk_active->execute([$puzzle_id, $cand]);
                if ($chk_active->fetchColumn()) continue;

                $chosen_block = $cand;
                break;
            }

            if ($chosen_block !== null) {
                $ins_blk->execute([$puzzle_id, $chosen_block, $now]);
                for ($i = 0; $i < RANGES_PER_BLOCK; $i++) {
                    $ins_rng->execute([$puzzle_id, $chosen_block, $i]);
                }
            }
        }
        $pdo->commit();
    } catch (Exception $e) {
        if ($pdo->inTransaction()) $pdo->rollBack();
    }
}

function format_speed(float $speed): string {
    if ($speed >= 1000000) return number_format($speed / 1000000, 2) . ' Mkeys/s';
    if ($speed >= 1000) return number_format($speed / 1000, 2) . ' Kkeys/s';
    return number_format($speed, 0) . ' keys/s';
}

function time_ago(int $ts): string {
    $diff = time() - $ts;
    if ($diff < 5) return 'just now';
    if ($diff < 60) return $diff . 's ago';
    $m = floor($diff / 60);
    return $m . 'm ago';
}

function format_eta(float $seconds): string {
    if ($seconds <= 0) return '0s';
    $days = floor($seconds / 86400);
    $years = floor($days / 365.25);
    $rem_days = (int)($days % 365);
    $hours = floor(($seconds % 86400) / 3600);
    $minutes = floor(($seconds % 3600) / 60);

    if ($years >= 100) {
        return number_format($years) . ' years';
    }
    if ($years > 0) {
        return $years . 'y ' . $rem_days . 'd';
    }
    if ($days > 0) {
        return $days . 'd ' . $hours . 'h';
    }
    if ($hours > 0) {
        return $hours . 'h ' . $minutes . 'm';
    }
    return max(1, $minutes) . 'm';
}

function calculate_probability_and_eta(int $puzzle_id, array $config, int $blocks_done, int $done_ranges_in_block, float $cluster_speed, bool $is_solved): array {
    $power = (int)($config['power_of_2'] ?? 70);
    $total_keys = pow(2, $power);

    $scanned_ranges = ($blocks_done * RANGES_PER_BLOCK) + $done_ranges_in_block;
    $keys_scanned = (float)$scanned_ranges * (float)RANGE_SIZE;
    $keys_remaining = max(0.0, $total_keys - $keys_scanned);

    $pct = ($keys_scanned / $total_keys) * 100.0;
    $pct_str = sprintf('%.8f%%', $pct);

    if ($is_solved) {
        return [
            'odds_str'        => '100% (TARGET HIT)',
            'odds_24h'        => 'Target completed',
            'probability_pct' => '100.00%',
            'keys_checked'    => 'Target solved',
            'eta_100_str'     => 'Completed (Solved)',
            'eta_100_seconds' => 0,
            'progress_pct'    => '100.00%',
        ];
    }

    $log10_total = $power * log10(2);
    $log10_scanned = ($keys_scanned > 0) ? log10($keys_scanned) : 0;

    if ($keys_scanned <= 0) {
        $odds_str = 'Starting...';
    } else {
        $diff_log = $log10_total - $log10_scanned;
        if ($diff_log <= 0) {
            $odds_str = '1 in 1 (100%)';
        } else {
            $odds_val = pow(10, $diff_log);
            if ($odds_val >= 1e12) {
                $odds_str = '1 in ' . number_format($odds_val / 1e12, 2) . ' trillion';
            } elseif ($odds_val >= 1e9) {
                $odds_str = '1 in ' . number_format($odds_val / 1e9, 2) . ' billion';
            } elseif ($odds_val >= 1e6) {
                $odds_str = '1 in ' . number_format($odds_val / 1e6, 2) . ' million';
            } elseif ($odds_val >= 1e3) {
                $odds_str = '1 in ' . number_format($odds_val / 1e3, 1) . ' thousand';
            } else {
                $odds_str = '1 in ' . number_format($odds_val, 0);
            }
        }
    }

    if ($cluster_speed > 0) {
        $keys_24h = $cluster_speed * 86400.0;
        $diff_log_24h = $log10_total - log10($keys_24h);
        if ($diff_log_24h <= 0) {
            $odds_24h = 'Estimated 24h: 1 in 1 / 24h';
        } else {
            $val_24h = pow(10, $diff_log_24h);
            if ($val_24h >= 1e12) {
                $odds_24h = 'Estimated 24h: 1 in ' . number_format($val_24h / 1e12, 2) . ' trillion';
            } elseif ($val_24h >= 1e9) {
                $odds_24h = 'Estimated 24h: 1 in ' . number_format($val_24h / 1e9, 2) . ' billion';
            } elseif ($val_24h >= 1e6) {
                $odds_24h = 'Estimated 24h: 1 in ' . number_format($val_24h / 1e6, 2) . ' million';
            } else {
                $odds_24h = 'Estimated 24h: 1 in ' . number_format($val_24h, 0);
            }
        }

        $eta_seconds = $keys_remaining / $cluster_speed;
        $eta_str = format_eta($eta_seconds);
    } else {
        $odds_24h = 'Estimated 24h: Waiting for hashrate...';
        $eta_seconds = 0;
        $eta_str = 'Waiting for hashrate...';
    }

    if ($keys_scanned >= 1e12) {
        $keys_fmt = number_format($keys_scanned / 1e12, 2) . ' trillion keys';
    } elseif ($keys_scanned >= 1e9) {
        $keys_fmt = number_format($keys_scanned / 1e9, 2) . ' billion keys';
    } elseif ($keys_scanned >= 1e6) {
        $keys_fmt = number_format($keys_scanned / 1e6, 2) . ' million keys';
    } else {
        $keys_fmt = number_format($keys_scanned, 0) . ' keys';
    }

    return [
        'odds_str'        => $odds_str,
        'odds_24h'        => $odds_24h,
        'probability_pct' => $pct_str,
        'keys_checked'    => $keys_fmt,
        'eta_100_str'     => $eta_str,
        'eta_100_seconds' => $eta_seconds,
        'progress_pct'    => $pct_str,
    ];
}

function render_html_dashboard(int $puzzle_id, array $config, array $stat, array $workers, int $blocks_done, float $total_speed, array $found_keys, array $prob_info, ?array $target_hit_info = null): void {
    header('Content-Type: text/html; charset=utf-8');
    $speed_str = format_speed($total_speed);
    $active_workers_count = count($workers);
    $done_ranges_str = number_format($stat['done_count'] ?? 0);
    $todo_ranges_str = number_format($stat['todo_count'] ?? 0);
    $active_ranges_str = number_format($stat['active_count'] ?? 0);
    $blocks_done_str = number_format($blocks_done);
    $total_blocks = (int)$config['total_blocks'];
    $total_blocks_str = number_format($total_blocks);
    $target_address = htmlspecialchars($config['target_address']);

    $puzzles_tabs_html = '';
    foreach (PUZZLES as $pid => $pconf) {
        $cls = ($pid === $puzzle_id) ? 'tab active' : 'tab';
        $label = 'Puzzle #' . $pid . ($pid === DEFAULT_PUZZLE ? ' (Main Target)' : ($pid === 70 ? ' (Test Vector)' : ''));
        $puzzles_tabs_html .= '<a href="?puzzle=' . $pid . '" class="' . $cls . '">' . $label . '</a>';
    }

    $found_banner_html = '';
    if (!empty($found_keys)) {
        $found_banner_html = '<div class="hit-alert-banner" id="hit-banner">'
            . '<div class="hit-badge"><span class="hit-sparkle">⚡</span> TARGET HIT CONFIRMED <span class="hit-sparkle">⚡</span></div>'
            . '</div>';
    }

    $rows_html = '';
    if (empty($workers)) {
        $rows_html = '<tr><td colspan="8" style="text-align: center; color: #64748b; padding: 24px;">No workers connected in the last 5 minutes. Run the worker below to start.</td></tr>';
    } else {
        foreach ($workers as $idx => $w) {
            $w_name = htmlspecialchars($w['worker'] ?? 'anonymous');
            $w_speed = format_speed((float)($w['speed'] ?? 0));
            $w_block = isset($w['current_block']) && $w['current_block'] > 0 ? '#' . number_format((int)$w['current_block']) : '-';
            $w_range = isset($w['current_range']) && $w['current_range'] >= 0 ? '#' . number_format((int)$w['current_range']) : '-';
            $w_done = number_format((int)($w['ranges_done'] ?? 0));
            $w_seen = time_ago((int)($w['last_seen'] ?? 0));
            $is_online = (time() - (int)($w['last_seen'] ?? 0)) <= 300;
            $badge = $is_online 
                ? '<span class="badge-online">● Online</span>' 
                : '<span class="badge-idle">○ Idle</span>';
            $rows_html .= '<tr>'
                . '<td>' . ($idx + 1) . '</td>'
                . '<td style="font-weight: 600; color: #f8fafc;">' . $w_name . '</td>'
                . '<td style="font-family: monospace; color: #38bdf8;">' . $w_speed . '</td>'
                . '<td style="font-family: monospace; color: #fbbf24;">' . $w_block . '</td>'
                . '<td style="font-family: monospace; color: #a78bfa;">' . $w_range . '</td>'
                . '<td style="font-family: monospace;">' . $w_done . '</td>'
                . '<td style="color: #94a3b8;">' . $w_seen . '</td>'
                . '<td>' . $badge . '</td>'
                . '</tr>';
        }
    }

    echo '<!DOCTYPE html>'
        . '<html lang="en">'
        . '<head>'
        . '<meta charset="utf-8">'
        . '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        . '<title>BTC Puzzle Cluster Dashboard - Puzzle #' . $puzzle_id . '</title>'
        . '<style>'
        . '* { box-sizing: border-box; }'
        . 'body { margin: 0; padding: 20px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0b0f19; color: #f1f5f9; line-height: 1.5; }'
        . '.container { max-width: 1200px; margin: 0 auto; }'
        . '.header { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid #1e293b; }'
        . '.brand { display: flex; align-items: center; gap: 10px; font-size: 20px; font-weight: 700; color: #38bdf8; }'
        . '.status-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 9999px; color: #34d399; font-size: 12px; font-weight: 600; }'
        . '.tabs { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }'
        . '.tab { padding: 8px 16px; border-radius: 8px; background: #1e293b; color: #94a3b8; text-decoration: none; font-size: 13px; font-weight: 600; border: 1px solid #334155; transition: all 0.2s; }'
        . '.tab.active { background: #0284c7; color: #ffffff; border-color: #38bdf8; }'
        . '.tab:hover:not(.active) { background: #334155; color: #f8fafc; }'
        . '.info-card { background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 18px 24px; margin-bottom: 20px; }'
        . '.grid-6 { display: grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap: 16px; margin-bottom: 25px; }'
        . '.stat-card { background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 18px; }'
        . '.stat-title { font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; margin-bottom: 8px; }'
        . '.stat-value { font-size: 22px; font-weight: 700; color: #f8fafc; font-family: monospace; }'
        . '.stat-sub { font-size: 12px; color: #64748b; margin-top: 6px; }'
        . '.hit-alert-banner { position: relative; background: radial-gradient(circle at center, rgba(16, 185, 129, 0.22) 0%, rgba(245, 158, 11, 0.16) 60%, rgba(15, 23, 42, 0.95) 100%); border: 2px solid #10b981; box-shadow: 0 0 35px rgba(16, 185, 129, 0.4), inset 0 0 25px rgba(245, 158, 11, 0.2); border-radius: 14px; padding: 24px 20px; margin-bottom: 25px; text-align: center; overflow: hidden; }'
        . '.hit-badge { display: inline-flex; align-items: center; gap: 8px; padding: 6px 18px; background: rgba(16, 185, 129, 0.25); border: 1px solid #34d399; border-radius: 9999px; color: #6ee7b7; font-size: 13px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 10px; }'
        . '.hit-sparkle { color: #fbbf24; }'
        . 'table { width: 100%; border-collapse: collapse; text-align: left; }'
        . 'th { padding: 12px 14px; font-size: 11px; font-weight: 600; color: #94a3b8; border-bottom: 1px solid #1e293b; text-transform: uppercase; background: #0f172a; white-space: nowrap; }'
        . 'td { padding: 12px 14px; font-size: 13px; border-bottom: 1px solid #1e293b; color: #cbd5e1; white-space: nowrap; }'
        . 'tr:hover td { background: rgba(30, 41, 59, 0.5); }'
        . '.badge-online { background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }'
        . '.badge-idle { background: rgba(245, 158, 11, 0.15); color: #fbbf24; padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }'
        . '.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981; animation: pulse 2s infinite; }'
        . '@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }'
        . '</style>'
        . '</head>'
        . '<body>'
        . '<div class="container">'
        . '<div class="header">'
        . '<div class="brand"><span>⚡</span> BTC Puzzle Cluster Dashboard</div>'
        . '<div style="display: flex; gap: 12px; align-items: center;">'
        . '<span class="status-badge"><span class="dot"></span> SQLite WAL Active</span>'
        . '<span id="live-indicator" style="font-size: 12px; color: #64748b;">Live Auto-Refresh (3s)</span>'
        . '</div>'
        . '</div>'
        . '<div class="tabs">' . $puzzles_tabs_html . '</div>'
        . '<div id="hit-banner-container">' . $found_banner_html . '</div>'
        . '<div class="info-card">'
        . '<div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 15px;">'
        . '<div>'
        . '<div style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Bitcoin Target Address (Puzzle #' . $puzzle_id . ')</div>'
        . '<div style="font-size: 18px; font-weight: 700; color: #f8fafc; font-family: monospace; margin-top: 4px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;">'
        . '<span>' . $target_address . '</span>'
        . '<span id="target-hit-badge" class="status-badge" style="' . (!empty($found_keys) ? '' : 'display:none;') . 'background: rgba(16,185,129,0.2); border-color: #10b981; color: #34d399;"><span class="dot"></span> SOLVED</span>'
        . '</div>'
        . '</div>'
        . '<div style="text-align: right;">'
        . '<div style="font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Total Blocks & Range Config</div>'
        . '<div style="font-size: 15px; font-weight: 600; color: #38bdf8; font-family: monospace; margin-top: 4px;">' . $total_blocks_str . ' Blocks | 268.4M keys/range (2^' . RANGE_SIZE_EXP . ')</div>'
        . '</div>'
        . '</div>'
        . '</div>'
        . '<div class="grid-6">'
        . '<div class="stat-card">'
        . '<div class="stat-title">⚡ Cluster Speed</div>'
        . '<div class="stat-value" id="val-speed" style="color: #34d399;">' . $speed_str . '</div>'
        . '<div class="stat-sub">From online workers</div>'
        . '</div>'
        . '<div class="stat-card">'
        . '<div class="stat-title">🎯 Probability (24h)</div>'
        . '<div class="stat-value" id="val-prob" style="color: #38bdf8; font-size: 19px;">' . htmlspecialchars($prob_info['odds_str']) . '</div>'
        . '<div class="stat-sub" id="val-prob-sub">' . htmlspecialchars($prob_info['odds_24h']) . '</div>'
        . '</div>'
        . '<div class="stat-card">'
        . '<div class="stat-title">⏳ Estimated to 100%</div>'
        . '<div class="stat-value" id="val-eta" style="color: #f59e0b; font-size: 19px;">' . htmlspecialchars($prob_info['eta_100_str']) . '</div>'
        . '<div class="stat-sub">Progress: <strong id="val-progress-pct" style="color: #38bdf8;">' . $prob_info['progress_pct'] . '</strong></div>'
        . '</div>'
        . '<div class="stat-card">'
        . '<div class="stat-title">🖥️ Active Workers</div>'
        . '<div class="stat-value" id="val-workers" style="color: #fbbf24;">' . $active_workers_count . ' Node(s)</div>'
        . '<div class="stat-sub">Reported in last 5m</div>'
        . '</div>'
        . '<div class="stat-card">'
        . '<div class="stat-title">📦 Total & Blocks Done</div>'
        . '<div class="stat-value" id="val-blocks" style="color: #a78bfa; font-size: 19px;">' . $blocks_done_str . ' / ' . $total_blocks_str . '</div>'
        . '<div class="stat-sub">Done ranges: <strong id="val-done">' . $done_ranges_str . '</strong></div>'
        . '</div>'
        . '<div class="stat-card">'
        . '<div class="stat-title">⏳ Ready Queue</div>'
        . '<div class="stat-value" id="val-todo" style="color: #f43f5e;">' . $todo_ranges_str . '</div>'
        . '<div class="stat-sub">Scanning: <strong id="val-active">' . $active_ranges_str . '</strong> ranges</div>'
        . '</div>'
        . '</div>'
        . '<div class="info-card" style="padding: 0; overflow: hidden; margin-bottom: 25px;">'
        . '<div style="padding: 16px 20px; border-bottom: 1px solid #1e293b; display: flex; justify-content: space-between; align-items: center;">'
        . '<span style="font-weight: 700; font-size: 14px; color: #f8fafc;">📋 Online Worker List & Block/Range Monitor</span>'
        . '<span style="font-size: 12px; color: #64748b;">Top 50 fastest nodes limit</span>'
        . '</div>'
        . '<div style="overflow-x: auto;">'
        . '<table>'
        . '<thead><tr><th>#</th><th>Worker Name</th><th>Speed</th><th>Block Scanning</th><th>Range Scanning</th><th>Ranges Submitted</th><th>Last Active</th><th>Status</th></tr></thead>'
        . '<tbody id="worker-tbody">' . $rows_html . '</tbody>'
        . '</table>'
        . '</div>'
        . '</div>'
        . '</div>'
        . '<script>'
        . 'var curPuzzle = ' . $puzzle_id . ';'
        . 'function formatSpeed(s) {'
        . '  if (s >= 1000000) return (s / 1000000).toFixed(2) + " Mkeys/s";'
        . '  if (s >= 1000) return (s / 1000).toFixed(2) + " Kkeys/s";'
        . '  return Math.round(s) + " keys/s";'
        . '}'
        . 'function timeAgo(ts) {'
        . '  var diff = Math.floor(Date.now() / 1000) - ts;'
        . '  if (diff < 5) return "just now";'
        . '  if (diff < 60) return diff + "s ago";'
        . '  return Math.floor(diff / 60) + "m ago";'
        . '}'
        . 'function refreshData() {'
        . '  fetch("?action=stats&puzzle=" + curPuzzle + "&format=json")'
        . '    .then(function(r) { return r.json(); })'
        . '    .then(function(data) {'
        . '      if (!data || data.status !== "ok") return;'
        . '      document.getElementById("val-speed").textContent = formatSpeed(data.cluster_speed || 0);'
        . '      document.getElementById("val-workers").textContent = (data.active_workers || 0) + " Node(s)";'
        . '      document.getElementById("val-done").textContent = (data.completed_ranges || 0).toLocaleString();'
        . '      document.getElementById("val-todo").textContent = (data.todo_ranges || 0).toLocaleString();'
        . '      document.getElementById("val-active").textContent = (data.active_ranges || 0).toLocaleString();'
        . '      if (document.getElementById("val-blocks") && data.blocks_done !== undefined) {'
        . '        var totB = data.total_blocks ? Number(data.total_blocks).toLocaleString() : "-";'
        . '        document.getElementById("val-blocks").textContent = Number(data.blocks_done).toLocaleString() + " / " + totB;'
        . '      }'
        . '      if (data.probability) {'
        . '        var pEl = document.getElementById("val-prob");'
        . '        if (pEl) pEl.textContent = data.probability.odds_str || "-";'
        . '        var pSub = document.getElementById("val-prob-sub");'
        . '        if (pSub) pSub.textContent = data.probability.odds_24h || "";'
        . '        var etaEl = document.getElementById("val-eta");'
        . '        if (etaEl) etaEl.textContent = data.probability.eta_100_str || "-";'
        . '        var pctEl = document.getElementById("val-progress-pct");'
        . '        if (pctEl) pctEl.textContent = data.probability.progress_pct || "0.00%";'
        . '      }'
        . '      if (data.target_solved) {'
        . '        var b = document.getElementById("hit-banner-container");'
        . '        if (b && !b.innerHTML.trim()) {'
        . '          b.innerHTML = "<div class=\"hit-alert-banner\" id=\"hit-banner\">" +'
        . '            "<div class=\"hit-badge\"><span class=\"hit-sparkle\">⚡</span> TARGET HIT CONFIRMED <span class=\"hit-sparkle\">⚡</span></div>" +'
        . '            "</div>";'
        . '        }'
        . '        var badge = document.getElementById("target-hit-badge");'
        . '        if (badge) badge.style.display = "inline-flex";'
        . '      }'
        . '      var tbody = document.getElementById("worker-tbody");'
        . '      if (tbody && data.workers && data.workers.length > 0) {'
        . '        var h = "";'
        . '        var now = Math.floor(Date.now() / 1000);'
        . '        for (var i = 0; i < data.workers.length; i++) {'
        . '          var w = data.workers[i];'
        . '          var isOnline = (now - w.last_seen) <= 300;'
        . '          var badge = isOnline ? "<span class=\"badge-online\">● Online</span>" : "<span class=\"badge-idle\">○ Idle</span>";'
        . '          var bText = (w.current_block !== undefined && w.current_block !== null && w.current_block > 0) ? "#" + Number(w.current_block).toLocaleString() : "-";'
        . '          var rText = (w.current_range !== undefined && w.current_range !== null && w.current_range >= 0) ? "#" + Number(w.current_range).toLocaleString() : "-";'
        . '          h += "<tr><td>" + (i + 1) + "</td><td style=\"font-weight: 600; color: #f8fafc;\">" + w.worker + "</td><td style=\"font-family: monospace; color: #38bdf8;\">" + formatSpeed(w.speed || 0) + "</td><td style=\"font-family: monospace; color: #fbbf24;\">" + bText + "</td><td style=\"font-family: monospace; color: #a78bfa;\">" + rText + "</td><td style=\"font-family: monospace;\">" + (w.ranges_done || 0).toLocaleString() + "</td><td style=\"color: #94a3b8;\">" + timeAgo(w.last_seen) + "</td><td>" + badge + "</td></tr>";'
        . '        }'
        . '        tbody.innerHTML = h;'
        . '      }'
        . '    }).catch(function(e) {});'
        . '}'
        . 'setInterval(refreshData, 3000);'
        . '</script>'
        . '</body></html>';
    exit;
}

$action = $_GET['action'] ?? '';
$input = null;
if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    $raw_in = file_get_contents('php://input');
    $input = json_decode($raw_in, true);
    if (is_array($input) && isset($input['action'])) $action = (string)$input['action'];
}
$action = strtolower($action);

switch ($action) {
    case 'health':
        respond(['status' => 'ok', 'engine' => 'SQLite WAL (CPU Optimized)', 'time' => time()]);
        break;

    case 'dashboard':
    case 'stats':
    case '':
        $raw_puzzle = $_GET['puzzle'] ?? ($input['puzzle'] ?? null);
        $puzzle_id = resolve_puzzle_id($raw_puzzle);
        $config = PUZZLES[$puzzle_id];
        $pdo = get_puzzle_db($puzzle_id);

        try {
            $st_full = $pdo->prepare("
                SELECT block_id, COUNT(*) as d_cnt 
                FROM ranges 
                WHERE puzzle_id = ? AND status IN (2, 3) 
                GROUP BY block_id 
                HAVING d_cnt >= ?
            ");
            $st_full->execute([$puzzle_id, RANGES_PER_BLOCK]);
            $full_blocks = $st_full->fetchAll();
            foreach ($full_blocks as $fb) {
                compact_completed_block($pdo, $puzzle_id, (int)$fb['block_id']);
            }
        } catch (Exception $e) {}

        $now = time();
        $st = $pdo->prepare("
            SELECT 
                SUM(CASE WHEN status = 0 THEN 1 ELSE 0 END) as todo_count,
                SUM(CASE WHEN status = 1 AND claimed_at >= ? THEN 1 ELSE 0 END) as active_count,
                SUM(CASE WHEN status = 2 THEN 1 ELSE 0 END) as done_count
            FROM ranges WHERE puzzle_id = ?
        ");
        $st->execute([$now - LEASE_TIMEOUT_SECS, $puzzle_id]);
        $stat = $st->fetch() ?: ['todo_count' => 0, 'active_count' => 0, 'done_count' => 0];

        $workers = $pdo->query("SELECT worker, speed, ranges_done, current_block, current_range, last_seen FROM user_stats WHERE last_seen >= ($now - 600) ORDER BY speed DESC LIMIT 50")->fetchAll();

        $st_int = $pdo->prepare("SELECT SUM(end_block - start_block + 1) as blocks_done FROM done_intervals WHERE puzzle_id = ?");
        $st_int->execute([$puzzle_id]);
        $blocks_done = (int)$st_int->fetchColumn();

        $res_json_file = get_results_json_path($puzzle_id);
        $found_keys = file_exists($res_json_file) ? (json_decode(file_get_contents($res_json_file), true) ?: []) : [];

        $total_speed = 0.0;
        foreach ($workers as $w) {
            $total_speed += (float)($w['speed'] ?? 0);
        }

        $prob_info = calculate_probability_and_eta($puzzle_id, $config, $blocks_done, (int)$stat['done_count'], $total_speed, !empty($found_keys));

        $target_hit_info = null;
        if (!empty($found_keys)) {
            $latest = end($found_keys);
            $target_hit_info = [
                'worker'    => $latest['worker'] ?? 'anonymous',
                'speed'     => (float)($latest['speed'] ?? 0),
                'block'     => (int)($latest['block'] ?? 0),
                'range_idx' => (int)($latest['range_idx'] ?? 0),
                'time'      => $latest['time'] ?? date('Y-m-d H:i:s'),
            ];
        }

        $is_json = ($action === 'stats')
            || (isset($_GET['format']) && $_GET['format'] === 'json')
            || (isset($_SERVER['HTTP_ACCEPT']) && strpos($_SERVER['HTTP_ACCEPT'], 'application/json') !== false && strpos($_SERVER['HTTP_ACCEPT'], 'text/html') === false);

        if ($is_json) {
            respond([
                'status'           => 'ok',
                'puzzle'           => $puzzle_id,
                'default_puzzle'   => DEFAULT_PUZZLE,
                'target_address'   => $config['target_address'],
                'total_blocks'     => (int)$config['total_blocks'],
                'ranges_per_block' => RANGES_PER_BLOCK,
                'range_size'       => (int)RANGE_SIZE,
                'todo_ranges'      => (int)$stat['todo_count'],
                'active_ranges'    => (int)$stat['active_count'],
                'completed_ranges' => (int)$stat['done_count'],
                'blocks_done'      => $blocks_done,
                'cluster_speed'    => $total_speed,
                'active_workers'   => count($workers),
                'workers'          => $workers,
                'probability'      => $prob_info,
                'target_solved'    => !empty($found_keys),
                'target_hit_info'  => $target_hit_info,
            ]);
        }

        render_html_dashboard($puzzle_id, $config, $stat, $workers, $blocks_done, $total_speed, $found_keys, $prob_info, $target_hit_info);
        break;

    case 'config':
        $raw_puzzle = $_GET['puzzle'] ?? null;
        $puzzle_id = resolve_puzzle_id($raw_puzzle);
        $config = PUZZLES[$puzzle_id];
        respond([
            'status'            => 'ok',
            'active_puzzle'     => $puzzle_id,
            'default_puzzle'    => DEFAULT_PUZZLE,
            'target_address'    => $config['target_address'],
            'total_blocks'      => (int)$config['total_blocks'],
            'range_size'        => (int)RANGE_SIZE,
            'range_size_exp'    => RANGE_SIZE_EXP,
            'ranges_per_block'  => RANGES_PER_BLOCK,
            'supported_puzzles' => array_keys(PUZZLES),
        ]);
        break;

    case 'range':
        if ($_SERVER['REQUEST_METHOD'] !== 'GET') error_resp('Method not allowed', 405);
        $raw_puzzle = $_GET['puzzle'] ?? null;
        // YÊU CẦU 4: nếu ko truyền puzzle thì hard code trả về puzzle hiện tại đang cần xử lý là 71
        $puzzle_id = ($raw_puzzle === null || $raw_puzzle === '' || !is_numeric($raw_puzzle)) ? 71 : resolve_puzzle_id($raw_puzzle);
        $config = PUZZLES[$puzzle_id];

        if ($puzzle_id === 70 || (isset($_GET['test']) && $_GET['test'] == 1)) {
            $test_block = 1382945497;
            $test_range = 49;
            $test_user  = !empty($_GET['user']) ? trim((string)$_GET['user']) : "user-{$test_block}-{$test_range}";
            respond([
                'status'           => 'ok',
                'puzzle'           => 70,
                'user'             => $test_user,
                'block'            => $test_block,
                'range_idx'        => $test_range,
                'start'            => '970436974004848820224',
                'end'              => '970436974005117255680',
                'range_size'       => (int)RANGE_SIZE,
                'target_address'   => '19YZECXj3SxEZMoUeJ1yiPsw8xANe7M7QR',
                'version_byte'     => 0,
                'lower'            => '590295810358705651712',
                'total'            => '590295810358705651712',
                'total_blocks'     => 2147483648,
                'ranges_per_block' => RANGES_PER_BLOCK,
                'is_test'          => true,
            ]);
        }

        $pdo = get_puzzle_db($puzzle_id);
        ensure_range_buffer($pdo, $puzzle_id, $config);

        $now = time();
        $expiry = $now - LEASE_TIMEOUT_SECS;

        $pdo->beginTransaction();
        try {
            // Ưu tiên dứt điểm từng block: block_id ASC, range_idx ASC
            $stmt = $pdo->prepare("
                SELECT block_id, range_idx 
                FROM ranges 
                WHERE puzzle_id = ? AND (status = 0 OR (status = 1 AND claimed_at < ?))
                ORDER BY block_id ASC, range_idx ASC
                LIMIT 1
            ");
            $stmt->execute([$puzzle_id, $expiry]);
            $claimed = $stmt->fetch();

            if (!$claimed) {
                $pdo->commit();
                respond(['status' => 'no_work', 'message' => 'Loading new blocks, try again in 2 seconds'], 200);
            }

            $b_id  = (int)$claimed['block_id'];
            $r_idx = (int)$claimed['range_idx'];

            // YÊU CẦU 4: random user id trả về, có thể lấy user-{blockid}-{range-id}
            if (!empty($_GET['user'])) {
                $worker = trim((string)$_GET['user']);
            } else {
                $worker = "user-{$b_id}-{$r_idx}";
            }

            $pdo->prepare("UPDATE ranges SET status = 1, worker = ?, claimed_at = ? WHERE puzzle_id = ? AND block_id = ? AND range_idx = ?")
                ->execute([$worker, $now, $puzzle_id, $b_id, $r_idx]);

            $pdo->prepare("
                INSERT INTO user_stats (worker, speed, ranges_done, current_block, current_range, last_seen)
                VALUES (?, 0, 0, ?, ?, ?)
                ON CONFLICT(worker) DO UPDATE SET
                    current_block = excluded.current_block,
                    current_range = excluded.current_range,
                    last_seen = excluded.last_seen
            ")->execute([$worker, $b_id, $r_idx, $now]);

            $pdo->commit();

            $lower = (string)$config['lower'];
            $block_keys = bcmul((string)RANGES_PER_BLOCK, RANGE_SIZE);
            $block_offset = bcmul((string)$b_id, $block_keys);
            $block_start = bcadd($lower, $block_offset);

            $range_offset = bcmul((string)$r_idx, RANGE_SIZE);
            $start = bcadd($block_start, $range_offset);
            $end = bcadd($start, RANGE_SIZE);

            // Trả về puzzle (71) và user (user-{blockid}-{range-id}) cho client
            respond([
                'puzzle'           => $puzzle_id,
                'user'             => $worker,
                'block'            => $b_id,
                'range_idx'        => $r_idx,
                'start'            => $start,
                'end'              => $end,
                'range_size'       => (int)RANGE_SIZE,
                'target_address'   => $config['target_address'],
                'version_byte'     => (int)$config['version_byte'],
                'lower'            => $lower,
                'total'            => (string)$config['total'],
                'total_blocks'     => (int)$config['total_blocks'],
                'ranges_per_block' => RANGES_PER_BLOCK,
            ]);
        } catch (Exception $e) {
            if ($pdo->inTransaction()) $pdo->rollBack();
            error_resp('Database busy: ' . $e->getMessage(), 500);
        }
        break;

    case 'result':
        if ($_SERVER['REQUEST_METHOD'] !== 'POST') error_resp('Method not allowed', 405);
        if (!is_array($input)) {
            $raw = file_get_contents('php://input');
            $input = json_decode($raw, true);
        }
        if (!is_array($input)) error_resp('Invalid JSON');

        try {
            $raw_puzzle = $input['puzzle'] ?? ($_GET['puzzle'] ?? null);
            $puzzle_id = resolve_puzzle_id($raw_puzzle);
            $pdo = get_puzzle_db($puzzle_id);

            $block_id  = (int)($input['block'] ?? -1);
            $range_idx = (int)($input['range_idx'] ?? -1);
            $status    = strtolower((string)($input['status'] ?? 'done'));
            $worker    = trim((string)($input['user'] ?? 'anonymous'));
            $speed     = (float)($input['speed'] ?? 0.0);

            if ($block_id < 0 || $range_idx < 0) error_resp('Missing block or range_idx');

            if ($status === 'found') {
                $priv_key = (string)($input['private_key'] ?? '');
                if (empty($priv_key)) error_resp('private_key is required when found');

                $found_record = [
                    'puzzle'      => $puzzle_id,
                    'block'       => $block_id,
                    'range_idx'   => $range_idx,
                    'private_key' => $priv_key,
                    'worker'      => $worker,
                    'speed'       => $speed,
                    'time'        => date('c'),
                ];

                $res_json = get_results_json_path($puzzle_id);
                $res_txt  = get_result_txt_path($puzzle_id);

                $results = file_exists($res_json) ? json_decode(file_get_contents($res_json), true) : [];
                $results[] = $found_record;
                file_put_contents($res_json, json_encode($results, JSON_PRETTY_PRINT));
                file_put_contents($res_txt, json_encode($found_record) . "\n", FILE_APPEND | LOCK_EX);

                $pdo->prepare("UPDATE ranges SET status = 3 WHERE puzzle_id = ? AND block_id = ? AND range_idx = ?")
                    ->execute([$puzzle_id, $block_id, $range_idx]);
            } else {
                $pdo->prepare("UPDATE ranges SET status = 2 WHERE puzzle_id = ? AND block_id = ? AND range_idx = ?")
                    ->execute([$puzzle_id, $block_id, $range_idx]);

                $st_chk = $pdo->prepare("SELECT COUNT(*) FROM ranges WHERE puzzle_id = ? AND block_id = ? AND status IN (2, 3)");
                $st_chk->execute([$puzzle_id, $block_id]);
                $cur_done = (int)$st_chk->fetchColumn();

                $pdo->prepare("UPDATE blocks SET done_count = ? WHERE puzzle_id = ? AND block_id = ?")
                    ->execute([$cur_done, $puzzle_id, $block_id]);

                if ($cur_done >= RANGES_PER_BLOCK) {
                    compact_completed_block($pdo, $puzzle_id, $block_id);
                }
            }

            $now = time();
            $pdo->prepare("
                INSERT INTO user_stats (worker, speed, ranges_done, current_block, current_range, last_seen)
                VALUES (?, ?, 1, ?, ?, ?)
                ON CONFLICT(worker) DO UPDATE SET
                    speed = excluded.speed,
                    ranges_done = ranges_done + 1,
                    current_block = excluded.current_block,
                    current_range = excluded.current_range,
                    last_seen = excluded.last_seen
            ")->execute([$worker, $speed, $block_id, $range_idx, $now]);

            respond(['status' => 'ok', 'puzzle' => $puzzle_id, 'block' => $block_id, 'range_idx' => $range_idx]);
        } catch (\Throwable $e) {
            error_resp('Database error: ' . $e->getMessage(), 500);
        }
        break;

    default:
        if (isset($_SERVER['HTTP_ACCEPT']) && strpos($_SERVER['HTTP_ACCEPT'], 'text/html') !== false) {
            header('Location: ' . strtok($_SERVER['REQUEST_URI'], '?'));
            exit;
        }
        respond(['status' => 'ok', 'engine' => 'SQLite WAL (CPU Optimized)']);
}
