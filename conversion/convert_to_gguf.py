"""Delegate GGUF serialization to llama.cpp's converter."""
import argparse,subprocess
def main():
 p=argparse.ArgumentParser();p.add_argument("checkpoint");p.add_argument("output");p.add_argument("--converter",required=True);p.add_argument("--outtype",default="f16");a=p.parse_args();subprocess.run([a.converter,a.checkpoint,"--outfile",a.output,"--outtype",a.outtype],check=True)
if __name__=="__main__": main()
