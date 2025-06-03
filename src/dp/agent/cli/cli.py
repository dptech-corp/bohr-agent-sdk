import os
import click
import subprocess
import sys
import shutil
from pathlib import Path
import signal
import uuid
import requests

from ..server.storage import storage_dict

@click.group()
def cli():
    """DP Agent CLI tool for managing science agent tasks."""
    pass

@cli.group()
def fetch():
    """Fetch resources for the science agent."""
    pass

@fetch.command()
@click.option('--type', 
              default='calculation',
              type=click.Choice(['cloud', 'calculation', 'device'], case_sensitive=False),
              help='Scaffolding type (cloud/calculation/device)')
def scaffolding(type):
    """Fetch scaffolding for the science agent."""
    click.echo(f"Generating {type} project scaffold...")
    
    # 获取模板目录路径
    templates_dir = Path(__file__).parent / 'templates'
    
    # 获取用户当前工作目录
    current_dir = Path.cwd()
    
    #判断需要创建哪种类型的文件夹
    if type == 'cloud':
        dir_name = current_dir / 'cloud'
    elif type == 'device':
        dir_name = current_dir / 'device'
    elif type == 'calculation':
        dir_name = current_dir / 'calculation'
        
    #判断文件夹是否存在，如果不存在则创建
    if dir_name.exists():
            click.echo(f"Warning: {dir_name} already exists，skipping...")
            click.echo(f"If you want to create a new scaffold, please delete the existing project folder first.")
    else:
        dir_name.mkdir(parents=True, exist_ok=True)
        
    #在目标文件夹中创建init文件
    init_file = current_dir / dir_name / '__init__.py'
    if not init_file.exists():
        init_file.write_text('')
        
    #从templates创建main.py文件
    main_template = templates_dir / 'main.py.template'
    main_file = current_dir / 'main.py'
    if not main_file.exists():
        shutil.copy2(main_template, main_file)
        
    #如果是device则从模板创建tescan_device.py文件
    if type == 'device':
        tescan_template = templates_dir / 'device' / 'tescan_device.py.template'
        tescan_file = current_dir / 'device' / 'tescan_device.py'
        if not tescan_file.exists():
            shutil.copy2(tescan_template, tescan_file)
            click.echo("\nCreated TescanDevice example implementation in device/tescan_device.py")
            click.echo("Please modify this file according to your actual device control requirements.")
    elif type == 'calculation':
        calculation_template = templates_dir / 'calculation' / 'simple.py.template'
        calculation_file = current_dir / 'calculation' / 'simple.py'
        if not calculation_file.exists():
            shutil.copy2(calculation_template, calculation_file)
            click.echo("\nCreated calculation example implementation in calculation/calculation.py")
            click.echo("Please modify this file according to your actual calculation control requirements.")
            
    click.echo("\nSucceed for fetching scaffold!")
    click.echo("Now you can use dp-agent run tool , dp-agent run agent or dp-agent run debug to run this project!")
    # 创建必要的目录结构
    # project_dirs = ['cloud', 'device','calculation']
    # for dir_name in project_dirs:
    #     dst_dir = current_dir / dir_name
        
    #     if dst_dir.exists():
    #         click.echo(f"Warning: {dir_name} already exists，skipping...")
    #         click.echo(f"If you want to create a new scaffold, please delete the existing project folder first.")
    #         continue
            
    #     # 只创建目录，不复制SDK文件
    #     dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建__init__.py文件以使目录成为Python包
    # for dir_name in project_dirs:
    #     init_file = current_dir / dir_name / '__init__.py'
    #     if not init_file.exists():
    #         init_file.write_text('')
    
    # # 从模板创建main.py文件
    # main_template = templates_dir / 'main.py.template'
    # main_file = current_dir / 'main.py'
    # if not main_file.exists():
    #     shutil.copy2(main_template, main_file)
    
    # # 从模板创建tescan_device.py文件
    # tescan_template = templates_dir / 'device' / 'tescan_device.py.template'
    # tescan_file = current_dir / 'device' / 'tescan_device.py'
    # if not tescan_file.exists():
    #     shutil.copy2(tescan_template, tescan_file)
    #     click.echo("\nCreated TescanDevice example implementation in device/tescan_device.py")
    #     click.echo("Please modify this file according to your actual device control requirements.")
        
    
    
    # calculation_template = templates_dir / 'calculation' / 'simple.py.template'
    # calculation_file = current_dir / 'calculation' / 'simple.py'
    # if not calculation_file.exists():
    #     shutil.copy2(calculation_template, calculation_file)
    #     click.echo("\nCreated calculation example implementation in calculation/calculation.py")
    #     click.echo("Please modify this file according to your actual calculation control requirements.")
    
    
    
    
    # click.echo("\nSucceed for fetching scaffold!")
    # click.echo("Now you can use dp-agent run-cloud or dp-agent run-device to run this project!")
        
@fetch.command()
def config():
    """Fetch configuration files for the science agent.
    
    Downloads .env file and replaces dynamic variables like MQTT_DEVICE_ID.
    Note: This command is only available in internal network environments.
    """
    click.echo("Fetching configuration...")
    
    # # 获取模板目录路径
    # templates_dir = Path(__file__).parent / 'templates'
    # current_dir = Path.cwd()
    
    # # 复制环境配置模板
    # env_template = templates_dir / '.env.template'
    # env_file = current_dir / '.env'
    
    remote_env_url = 'https://openfiles.mlops.dp.tech/projects/bohrium-agent/69b192c8c9774a8b860ecb174a218c0b/env-public'
    env_file = Path.cwd() / '.env'
    
    if env_file.exists():
        click.echo("Warning: .env file already exists. Skipping...")
        click.echo("If you want to create a new .env file, please delete the existing one first.")
    else:
        response = requests.get(remote_env_url)
        with open(env_file, 'w') as f:
            f.write(response.text)
        #shutil.copy2(env_template, env_file)
        click.echo("Configuration file .env has been created.")
        click.echo("\nIMPORTANT: Please update the following configurations in your .env file:")
        click.echo("1. MQTT_INSTANCE_ID - Your Aliyun MQTT instance ID")
        click.echo("2. MQTT_ENDPOINT - Your Aliyun MQTT endpoint")
        click.echo("3. MQTT_DEVICE_ID - Your device ID")
        click.echo("4. MQTT_GROUP_ID - Your group ID")
        click.echo("5. MQTT_AK - Your Access Key")
        click.echo("6. MQTT_SK - Your Secret Key")
        click.echo("\nFor local development, you may also need to update:")
        click.echo("- MQTT_BROKER and MQTT_PORT")
        click.echo("- TESCAN_API_BASE")
    
    click.echo("\nConfiguration setup completed.")

@cli.group()
def run():
    """Run the science agent in different environments."""
    pass

@run.group()
def tool():
    """Run specific tool environment."""
    pass

@tool.command()
def cloud():
    """Run the science agent in cloud environment."""
    click.echo("Starting cloud environment...")
    
    try:
        subprocess.run([sys.executable, "main.py", "cloud"], check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"Run failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        click.echo("Error: main.py not found. Please run scaffolding command first.")
        sys.exit(1)

@tool.command()
def device():
    """Run the science agent in device environment."""
    click.echo("Starting device environment...")
    
    try:
        subprocess.run([sys.executable, "main.py", "device"], check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"Run failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        click.echo("Error: main.py not found. Please run scaffolding command first.")
        sys.exit(1)

@tool.command()
def calculation():
    """Run the science agent in calculation environment."""
    click.echo("Starting calculation environment...")
    
    try:
        subprocess.run([sys.executable, "main.py", "calculation"], check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"Run failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        click.echo("Error: main.py not found. Please run scaffolding command first.")
        sys.exit(1)

# @run.command()
# def lab():
#     """Run the science agent in lab environment."""
#     click.echo("Starting lab environment...")
    
#     try:
#         subprocess.run([sys.executable, "main.py", "lab"], check=True)
#     except subprocess.CalledProcessError as e:
#         click.echo(f"Run failed: {e}")
#         sys.exit(1)
#     except FileNotFoundError:
#         click.echo("Error: main.py not found. Please run scaffolding command first.")
#         sys.exit(1)

# @run.command()
# def cloud():
#     """Run the science agent in cloud environment."""
#     click.echo("Starting cloud environment...")
    
#     try:
#         subprocess.run([sys.executable, "main.py", "cloud"], check=True)
#     except subprocess.CalledProcessError as e:
#         click.echo(f"Run failed: {e}")
#         sys.exit(1)
#     except FileNotFoundError:
#         click.echo("Error: main.py not found. Please run scaffolding command first.")
#         sys.exit(1)

@run.command()
def agent():
    """Run the science agent."""
    click.echo("Starting agent...")
    click.echo("Agent started.")

@run.command()
def debug():
    """Debug the science agent in cloud environment."""
    click.echo("Starting cloud environment in debug mode...")
    click.echo("Cloud environment debug mode started.")

@cli.group()
def artifact():
    """Manipulate the artifacts."""
    pass

@artifact.command()
@click.argument("path")
@click.option("-p", "--prefix", default=None,
              help="Prefix in the artifact repository where the artifact uploaded to, 'upload/<uuid>' by default")
@click.option("-s", "--scheme", default=None, help="Storage scheme, 'local' by default")
def upload(**kwargs):
    """Upload a file/directory from local to artifact repository"""
    path = kwargs["path"]
    prefix = kwargs["prefix"]
    scheme = kwargs["scheme"]
    if prefix and "://" in prefix:
        offset = prefix.find("://")
        scheme = prefix[:offset]
        prefix = prefix[offset+3:]
    if scheme is None:
        scheme = "local"
    if prefix is None:
        prefix = "upload/%s" % uuid.uuid4()
    storage = storage_dict[scheme]()
    key = storage.upload(prefix, path)
    uri = "%s://%s" % (scheme, key)
    click.echo("%s has been uploaded to %s" % (path, uri))

@artifact.command()
@click.argument("uri")
@click.option("-p", "--path", default=".", help="Path where the artifact downloaded to, '.' by default")
def download(**kwargs):
    """Download an artifact from artifact repository to local"""
    uri = kwargs["uri"]
    path = kwargs["path"]
    if "://" in uri:
        offset = uri.find("://")
        scheme = uri[:offset]
        key = uri[offset+3:]
    else:
        scheme = "local"
        key = uri
    storage = storage_dict[scheme]()
    path = storage.download(key, path)
    click.echo("%s has been downloaded to %s" % (uri, path))

def main():
    cli()

if __name__ == "__main__":
    main()
