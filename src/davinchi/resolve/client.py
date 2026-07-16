"""
Connection handling for DaVinci Resolve's Python scripting API.

IMPORTANT — this module is UNTESTED against a real Resolve installation
(there's no Resolve app in the environment this was built in). It follows
Blackmagic's documented scripting API, but you'll need to verify paths and
behavior on your machine. The two most common gotchas:

1. Resolve must be running, with External Scripting enabled:
   Preferences > General > External scripting using = "Local" (or "Network").
2. The scripting modules aren't pip-installable — they live inside the
   Resolve install and need to be on sys.path / importable. Typical
   locations (verify against your actual install):

   Windows:
     %PROGRAMDATA%\\Blackmagic Design\\DaVinci Resolve\\Support\\Developer\\Scripting\\Modules\\
   macOS:
     /Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules/
   Linux:
     /opt/resolve/Developer/Scripting/Modules/  (or /home/resolve/... on some installs)

   Also requires the RESOLVE_SCRIPT_API and RESOLVE_SCRIPT_LIB environment
   variables to be set per Blackmagic's docs — check the README.txt that
   ships alongside the Modules folder on your machine for the exact values.
"""
import sys
import os


def _ensure_resolve_on_path():
    """Add Resolve's scripting modules directory to sys.path if not already there."""
    candidates = [
        os.environ.get("RESOLVE_SCRIPT_API_MODULES"),
        r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules",
        "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules",
        "/opt/resolve/Developer/Scripting/Modules",
    ]
    for path in candidates:
        if path and os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)


def get_resolve():
    """Return the Resolve scripting app object, or raise with a clear message."""
    _ensure_resolve_on_path()
    try:
        import DaVinciResolveScript as dvr_script
    except ImportError as e:
        raise ImportError(
            "Could not import DaVinciResolveScript. Make sure Resolve is "
            "installed, External Scripting is enabled in Preferences, and "
            "the Modules directory is discoverable (see client.py docstring)."
        ) from e

    resolve = dvr_script.scriptapp("Resolve")
    if resolve is None:
        raise RuntimeError(
            "DaVinciResolveScript imported but scriptapp('Resolve') returned "
            "None — is Resolve actually running?"
        )
    return resolve


def get_current_project(resolve=None):
    resolve = resolve or get_resolve()
    project_manager = resolve.GetProjectManager()
    project = project_manager.GetCurrentProject()
    if project is None:
        raise RuntimeError("No project is currently open in Resolve.")
    return project


def get_current_timeline(project=None):
    project = project or get_current_project()
    timeline = project.GetCurrentTimeline()
    if timeline is None:
        raise RuntimeError("No timeline is currently open in the project.")
    return timeline
