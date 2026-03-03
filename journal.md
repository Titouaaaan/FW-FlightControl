# Personal Notes for the fork of [FW-FlightControl](https://github.com/Akkodis/FW-FlightControl)

## Progress report

I will try and keep this updated regularly to not forget the things i did/want to do. 
Might migrate this to another file to avoid adding unnecessary files to the repository.

### 02/03/2026

- First time checking out the repo, parsed through files, multiples things missing, couple bugs but will work on them tomorrow.

- Figured out how to install all dependencies, including FW-JSBSim

### 03/03/2026

- fixed bug where dif_obs has inhomogeneous shape and breaks when converting to array, so we pad, 
there used to be a fix but only for SAC so now the fix works for all agent types (see `train_utils.py`)

- imported version of TD-MPC2 is not up to date: it still uses `gym` instead of `gymnasium` which is deprecated.
Currently removed its import in the training script (needs to be done at other places too). Need to change the submodule pointer
to a more recent version, I will also import it myself to use it directly locally. 

- fixed couple other bugs, like occasional use of `envs.envs[0]` instead of `unwr_envs[0]` which caused errors, nothing too major

- ran a training run, obtained results similar to the logs shown in Appendix B. of Chapter 5 in the [thesis manuscript](https://theses.hal.science/tel-05467318v1), so hopefully nothing broken at the moment

- fixed another env bug in the `ppo_eval_simple.py` script, where a function needed to be called on the unwrapped env instead of wrapped one

#### Tasks identified

- [ ] test training script doesn't break for SAC (it does)

- [ ] import TD-MPC2 properly (fix submodule/imports)

- [ ] test TD-MPC2 training script 

#### Notes

Currently for PPO, I managed to train and evaluate a PPO agent and it achieves the same scores as the ones found in the paper. 
The scores are between -50 and -25 approximately which is what we observe on the graphs in the appendix.

The bugs I fixed are most lkely due to dependency version changes, so it was just not maintained for newer versions most likely.

My agent is saved but ignored, i used this coommand to evaluate it: `python eval/attitude_control/ppo_eval_simple.py rl.PPO.env_id=ACBohnNoVaIErr-v0 env/jsbsim=gustsonly model_path=models/train/ppo/2864/ppo_gustsonly_2864.pt res_file=ppo_gustsonly_1 ref_file=simple_easy`,
and to train it's the same one as in the README.md!

Started working on SAC, execute train with: `python train/sac_train.py rl.SAC.env_id=ACBohnNoVaIErr-v0 rl.SAC.exp_name=gustsonly env/jsbsim=gustsonly`
