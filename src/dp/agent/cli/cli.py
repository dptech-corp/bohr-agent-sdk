import os
import click
import subprocess
import sys
import shutil
from pathlib import Path
import signal

@click.group()
def cli():
    """DP Agent CLI tool for managing science agent tasks."""
    pass

@cli.group()
def fetch():
    """Fetch resources for the science agent."""
    pass

@fetch.command()
def scaffolding():
    """Fetch scaffolding for the science agent."""
    click.echo("Generating...")
    
    # 获取模板目录路径
    templates_dir = Path(__file__).parent / 'templates'
    
    # 获取用户当前工作目录
    current_dir = Path.cwd()
    
    # 创建必要的目录结构
    project_dirs = ['cloud', 'lab']
    for dir_name in project_dirs:
        dst_dir = current_dir / dir_name
        
        if dst_dir.exists():
            click.echo(f"Warning: {dir_name} already exists，skipping...")
            click.echo(f"If you want to create a new scaffold, please delete the existing project folder first.")
            continue
            
        # 只创建目录，不复制SDK文件
        dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建__init__.py文件以使目录成为Python包
    for dir_name in project_dirs:
        init_file = current_dir / dir_name / '__init__.py'
        if not init_file.exists():
            init_file.write_text('')
    
    # 从模板创建main.py文件
    main_template = templates_dir / 'main.py.template'
    main_file = current_dir / 'main.py'
    if not main_file.exists():
        shutil.copy2(main_template, main_file)
    
    # 从模板创建tescan_device.py文件
    tescan_template = templates_dir / 'lab' / 'tescan_device.py.template'
    tescan_file = current_dir / 'lab' / 'tescan_device.py'
    if not tescan_file.exists():
        shutil.copy2(tescan_template, tescan_file)
        click.echo("\nCreated TescanDevice example implementation in lab/tescan_device.py")
        click.echo("Please modify this file according to your actual device control requirements.")
        
    click.echo("\nSucceed for fetching scaffold!")
    click.echo("Now you can use dp-agent run-cloud or dp-agent run-lab to run this project!")

@fetch.command()
def config():
    """Fetch configuration files for the science agent.
    
    Downloads .env file and replaces dynamic variables like MQTT_DEVICE_ID.
    Note: This command is only available in internal network environments.
    """
    click.echo("Fetching configuration...")
    
    # 获取模板目录路径
    templates_dir = Path(__file__).parent / 'templates'
    current_dir = Path.cwd()
    
    # 复制环境配置模板
    env_template = templates_dir / '.env.template'
    env_file = current_dir / '.env'
    
    if env_file.exists():
        click.echo("Warning: .env file already exists. Skipping...")
        click.echo("If you want to create a new .env file, please delete the existing one first.")
    else:
        shutil.copy2(env_template, env_file)
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

@run.command()
def lab():
    """Run the science agent in lab environment."""
    click.echo("Starting lab environment...")
    
    try:
        subprocess.run([sys.executable, "main.py", "lab"], check=True)
    except subprocess.CalledProcessError as e:
        click.echo(f"Run failed: {e}")
        sys.exit(1)
    except FileNotFoundError:
        click.echo("Error: main.py not found. Please run scaffolding command first.")
        sys.exit(1)

@run.command()
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

def main():
    cli()

if __name__ == "__main__":
    main()
