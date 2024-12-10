import os
import subprocess
import argparse

def create_argparser():
    parser = argparse.ArgumentParser()
    # I/O
    parser.add_argument("--root_path", type=str, default="/YourInputPath", help='directory that contains the .glb models')
    # intrinsics
    parser.add_argument("--resolution", type=int, default=512, help="resolution of the rendering")
    parser.add_argument("--lens", type=int, default=35, help="focal length in mm")
    parser.add_argument("--sensor_size", type=int, default=32, help="focal length in mm")
    return parser

if __name__ == "__main__":
    args = create_argparser().parse_args()
    root_path = args.root_path
    model_name_list = sorted(os.listdir(root_path))

    scripts_to_run = []
    for model_name in model_name_list:
        model_path = os.path.join(root_path, model_name)
        render_script = [
            'python', 'bot2colmap.py',
            '--out_path', model_path,
            '--resolution', str(args.resolution),
            '--lens', str(args.lens),
            '--sensor_size', str(args.sensor_size)
        ]
        scripts_to_run.append(render_script)

for script in scripts_to_run:
    print("Command to run:", ' '.join(script))


for script in scripts_to_run:
    try:
        result = subprocess.run(script, check=True)
        print(f"Executed {script[1]} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while executing {script[1]}: {e}")
        break