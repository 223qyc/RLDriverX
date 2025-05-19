import numpy as np
import cv2
from environment import CarEnvironment
from config.environment_config import ENV_CONFIG

def test_environment():
    # 创建环境
    env = CarEnvironment()
    
    # 重置环境
    obs, info = env.reset()
    
    # 测试几个随机动作
    for _ in range(100):
        # 随机动作
        action = np.random.uniform(-1, 1, size=2)
        
        # 执行动作
        obs, reward, done, truncated, info = env.step(action)
        
        # 显示视觉观察
        cv2.imshow('Environment', obs['visual'])
        cv2.waitKey(50)  # 50ms延迟
        
        # 打印信息
        print(f"Reward: {reward:.2f}, Distance to target: {info['distance_to_target']:.2f}")
        
        if done:
            print("Episode finished!")
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_environment() 