"""Apply the narrow Modal/gVisor FreeVideo memory-guard compatibility patch.

Modal's gVisor exposes /proc/PID/status but can deny smaps_rollup.  RSS remains
available and is a conservative upper bound, so use it only when explicitly
enabled by the Modal image.  Upstream source mismatches fail the image build.
"""
from pathlib import Path

path = Path("/opt/freevideo/freevideo_engine/ram.py")
source = path.read_text(encoding="utf-8")

source = source.replace(
    "from pathlib import Path\nfrom .system import windows\n",
    "import os\nfrom pathlib import Path\nfrom .system import windows\n",
    1,
)

old = """        self.samples += 1
        return {'rss_bytes': rss, 'pss_bytes': pss if not incomplete else None,
                # Unlike the guard's upper bound, this excludes every file page.
                # Live planning can add it to available memory without counting
                # clean mapped cache (already reclaimable) a second time.
                'nonfile_pss_bytes': pss - file_pss if file_counters_complete and not incomplete else None,
                'inference_guard_bytes': inference_pss if not incomplete else None,
                'inference_file_protection_bytes': protected_file,
                'guard_bytes': pss if not incomplete else None, 'guard_metric': 'PSS',
"""

new = """        self.samples += 1
        # Modal's gVisor may expose process status/RSS while denying smaps_rollup.
        # RSS double-counts shared pages, making it a conservative guard fallback.
        # Keep this opt-in so other runtimes retain upstream's strict PSS policy.
        rss_fallback = bool(incomplete and os.environ.get('FREEVIDEO_ALLOW_RSS_GUARD') == '1')
        return {'rss_bytes': rss, 'pss_bytes': pss if not incomplete else None,
                # Unlike the guard's upper bound, this excludes every file page.
                # Live planning can add it to available memory without counting
                # clean mapped cache (already reclaimable) a second time.
                'nonfile_pss_bytes': pss - file_pss if file_counters_complete and not incomplete else None,
                'inference_guard_bytes': (rss if rss_fallback else
                                          inference_pss if not incomplete else None),
                'inference_file_protection_bytes': protected_file,
                'guard_bytes': (rss if rss_fallback else pss if not incomplete else None),
                'guard_metric': 'RSS-fallback' if rss_fallback else 'PSS',
"""

if old not in source:
    raise SystemExit("FreeVideo ram.py no longer matches the audited pinned source")
source = source.replace(old, new, 1)
path.write_text(source, encoding="utf-8")
