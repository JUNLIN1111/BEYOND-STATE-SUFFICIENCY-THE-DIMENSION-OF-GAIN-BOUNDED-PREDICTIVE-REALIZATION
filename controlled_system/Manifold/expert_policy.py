import metaworld
from metaworld.policies import SawyerReachV3Policy
import time

# 1. 创建 MT1 benchmark
mt1 = metaworld.MT1('reach-v3')

# 2. 拿到单任务 env
env_cls = mt1.train_classes['reach-v3']
env = env_cls()
env.render_mode = 'human'

task = mt1.train_tasks[0]
env.set_task(task)

# 4. reset
obs, _ = env.reset()

policy = SawyerReachV3Policy()

done = False
step = 0
while not done:
    a = policy.get_action(obs)
    obs, _, _, _, info = env.step(a)
    # if step == 10:
        # img = env.render()
        # print(f"img is {img}")
    done = info['success'] == 1
    step += 1
    time.sleep(0.2)  
env.close()
print(f"Final observation: {obs}")
print(f"Episode finished after {step} steps with success: {info['success']}")