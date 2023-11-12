import subprocess
import os

cwd = os.getcwd()
subprocess.call(["sh", cwd+"/train.sh"])