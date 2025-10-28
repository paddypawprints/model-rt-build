#!/usr/bin/env python3
"""
Quick TensorRT / environment diagnostic.

Save this file and run it in the same environment you run your TRT comparison script:
  python trt_diagnostic.py

It prints:
 - Python executable and version
 - first entries of sys.path
 - whether a local tensorrt.py or tensorrt/ directory exists in the current directory
 - pkgutil.find_loader('tensorrt')
 - tensorrt import attempt info (file, version, sample attributes)
 - pip show tensorrt / python3-libnvinfer info (if available)
 - trtexec version/check via typical system paths
 - dpkg query for installed nvinfer packages (Jetson / Debian)
This helps identify if the Python tensorrt bindings are the system ones or a mismatched/shadowed package.
"""
from __future__ import print_function
import sys
import os
import pkgutil
import traceback
import importlib
import subprocess

def run_cmd(cmd):
    try:
        p = subprocess.run(cmd, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out = p.stdout.strip()
        err = p.stderr.strip()
        return p.returncode, out, err
    except Exception as e:
        return 1, "", f"Exception running command: {e}"

def print_heading(h):
    print("\n" + "="*10 + " " + h + " " + "="*10)

def main():
    print_heading("Python / Environment")
    print("Python executable:", sys.executable)
    print("Python version:", sys.version.replace("\n", " "))
    print("First entries of sys.path:")
    for p in sys.path[:8]:
        print(" ", p)

    # Check for local shadowing tensorrt files/dirs
    print_heading("Local files that could shadow 'tensorrt' import")
    cwd_items = sorted(os.listdir("."))
    shadow = [n for n in cwd_items if n.lower().startswith("tensorrt")]
    if shadow:
        print("Found potential shadowing file(s)/dir(s) in CWD:")
        for s in shadow:
            print(" ", s)
    else:
        print("No tensorrt* file/dir in CWD.")

    print_heading("pkgutil / loader info")
    loader = pkgutil.find_loader("tensorrt")
    print("pkgutil.find_loader('tensorrt') ->", loader)

    # Try importing tensorrt and show attributes
    print_heading("Attempt to import tensorrt (Python)")
    try:
        import tensorrt as trt
        print("Imported tensorrt OK.")
        # C-extension modules may not have __file__, but print if available
        print("  tensorrt __file__:", getattr(trt, "__file__", "<C-extension or built-in>"))
        print("  tensorrt __version__:", getattr(trt, "__version__", "<no __version__>"))
        print("  Has Logger attribute?:", hasattr(trt, "Logger"))
        # print a short sample of public attributes
        public = [k for k in dir(trt) if not k.startswith("_")]
        print("  sample attributes:", public[:40])
    except Exception:
        print("Import tensorrt FAILED. Traceback:")
        traceback.print_exc()

    # Show pip-installed package info (if pip is available)
    print_heading("pip / package info for tensorrt")
    rc, out, err = run_cmd(f"{sys.executable} -m pip show tensorrt")
    if rc == 0 and out:
        print("pip show tensorrt:\n", out)
    else:
        print("pip show tensorrt: not found or error. stderr:", err or "<none>")

    # Show apt/dpkg info (useful on Jetson)
    print_heading("dpkg (nvinfer packages) - requires apt/dpkg (Jetson/Debian)")
    rc, out, err = run_cmd("dpkg -l | grep -i nvinfer || true")
    if out:
        print(out)
    else:
        print("No dpkg nvinfer lines found or dpkg not available. stderr:", err or "<none>")

    # Show common Python dist-packages locations for nvinfer files
    print_heading("Look for nvinfer / tensorrt files in common system locations")
    candidates = [
        "/usr/lib/python3/dist-packages",
        "/usr/lib/python3.10/dist-packages",
        "/usr/lib/python3.8/dist-packages",
        "/usr/local/lib/python3.10/dist-packages",
        "/usr/local/lib/python3.8/dist-packages",
    ]
    for c in candidates:
        try:
            if os.path.isdir(c):
                items = [x for x in os.listdir(c) if "nvinfer" in x.lower() or "tensorrt" in x.lower()]
                if items:
                    print(f" {c}:")
                    for it in items:
                        print("   ", it)
        except Exception:
            pass

    # Try to get trtexec information (system binary)
    print_heading("trtexec / TensorRT binary check")
    # prefer full path if present
    for cmd in ["/usr/src/tensorrt/bin/trtexec --version", "trtexec --version", "/usr/src/tensorrt/bin/trtexec --help | head -n 1 || true", "trtexec --help | head -n 1 || true"]:
        rc, out, err = run_cmd(cmd)
        if rc == 0 and out:
            print(f"$ {cmd}\n{out.splitlines()[0]}")
            break
    else:
        print("trtexec not found in common locations or returned no output. stderr (last attempt):", err or "<none>")

    # Optionally try to load engine via trtexec for a quick header (if engine exists)
    engine_path = "image_fp16.engine"
    if os.path.exists(engine_path):
        print_heading(f"trtexec inspect of {engine_path}")
        rc, out, err = run_cmd(f"/usr/src/tensorrt/bin/trtexec --loadEngine={engine_path} --verbose")
        if rc == 0:
            # print a concise excerpt that shows bindings/created message (avoid printing huge logs)
            lines = out.splitlines()
            excerpt = []
            for line in lines:
                if "Input binding" in line or "Output binding" in line or "Loaded engine size" in line or "Created execution context" in line:
                    excerpt.append(line)
            if excerpt:
                print("\n".join(excerpt))
            else:
                print("trtexec loaded engine but no concise binding lines found in output. (Full output omitted.)")
        else:
            print("trtexec inspect failed. stderr (first 400 chars):\n", (err or out)[:400])

    print_heading("Quick guidance")
    print(" - If pkgutil.find_loader shows a loader but tensorrt.__file__ points to a local path in your project, rename that file (e.g., tensorrt.py) and retry.")
    print(" - If Python can't import tensorrt or version mismatches trtexec, run the script with the system Python (e.g., /usr/bin/python3) or install the system TensorRT Python bindings into the active interpreter.")
    print(" - To share results, paste the full output of this script here.")
    print("\nDone.")

if __name__ == "__main__":
    main()