import os
import sys
import subprocess

def main():
    config_path = 'options/train/KLA/NAFNet-width32-FrozenEncoder-KLA.yml'
    
    # Parse positional or -opt arguments
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        if '-opt' in args:
            opt_idx = args.index('-opt')
            if opt_idx + 1 < len(args):
                config_path = args[opt_idx + 1]
        else:
            config_path = args[0]
    
    print(f"[+] Starting NAFNet training with configuration: {config_path}")
    cmd = [sys.executable, 'basicsr/train.py', '-opt', config_path]
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
