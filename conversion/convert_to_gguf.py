"""Export a NeuroMind checkpoint to GGUF using llama.cpp's converter.

This module intentionally delegates GGUF serialization to llama.cpp rather than
reimplementing the binary format. Install/build llama.cpp and provide its
convert_hf_to_gguf.py path with --converter.
"""
from __future__ import annotations
import argparse, subprocess

def main():
    p=argparse.ArgumentParser(); p.add_argument("checkpoint"); p.add_argument("output"); p.add_argument("--converter",required=True); p.add_argument("--outtype",default="f16"); a=p.parse_args()
    subprocess.run([a.converter,a.checkpoint,"--outfile",a.output,"--outtype",a.outtype],check=True)
if __name__=="__main__": main()
