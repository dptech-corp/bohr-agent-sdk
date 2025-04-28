from dp.agent.server import CalculationMCPServer
import subprocess
import os   
import time
mcp = CalculationMCPServer("Demo")

@mcp.tool()

def vasp_job() -> str:
    """提交VASP计算任务
        
    Returns:
        str: 提交成功返回任务ID，失败返回错误信息
        
    Raises:
        FileNotFoundError: 当目录不存在或无法访问时
        Exception: 复制文件或提交任务失败时
    """
    start_dir = os.getcwd()
    try:
        os.chdir('tmp')
        
        subdirs = [d for d in os.listdir() if os.path.isdir(d)]
        if not subdirs:
            raise Exception("tmp目录下没有子目录")
        
        for subdir in subdirs:
            print(f"开始在子目录 {subdir} 中执行VASP...")
            os.chdir(subdir)
            
            command = "source /opt/intel/oneapi/setvars.sh && mpirun -n 32 vasp_std > vasp.log 2>&1"
            subprocess.run(['bash', '-c', command], check=True)
            
            os.chdir('..')
            print(f"子目录 {subdir} 执行完成")
        
    except Exception as e:
        print(f"执行出错: {e}")
        raise
    finally:

        os.chdir(start_dir)
    return {"status": "success", "message": "任务提交成功🎉"}

if __name__ == "__main__":

    mcp.run(transport="sse")
