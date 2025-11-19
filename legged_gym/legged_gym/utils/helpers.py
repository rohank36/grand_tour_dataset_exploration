import os
import copy
import torch
import numpy as np
import random
from isaacgym import gymapi
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_sim_params(args, cfg):
    # code from Isaac Gym Preview 2
    # initialize sim params
    sim_params = gymapi.SimParams()

    # set some values from args
    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs

        # Dictionary mapping argument attributes to environment config attributes
        reward_mappings = {
            "feet_air_time": "rewards.scales.feet_air_time",
            "hip_abduction_adduction": "rewards.scales.hip_abduction_adduction",
            "max_leg_spread": "rewards.max_leg_spread",
            "foot_drag": "rewards.scales.foot_drag",
        }

        domain_rand_mappings = {
            "domain_rand_friction_range": ("domain_rand.friction_range", "domain_rand.randomize_friction", True),
            "domain_rand_added_mass_range": ("domain_rand.added_mass_range", "domain_rand.randomize_added_mass", True),
            "domain_rand_push_range": ("domain_rand.max_push_vel_xy", "domain_rand.push_robots", True),
            "domain_rand_ground_friction_range": ("domain_rand.ground_friction_range", "domain_rand.randomize_ground_friction", True),
        }

        # Set reward parameters
        for arg, attr in reward_mappings.items():
            if getattr(args, arg) is not None:
                set_nested_attr(env_cfg, attr, getattr(args, arg))

        # Set domain randomization parameters
        for arg, (attr, toggle_attr, toggle_val) in domain_rand_mappings.items():
            if getattr(args, arg) is not None:
                set_nested_attr(env_cfg, attr, getattr(args, arg))
                if isinstance(toggle_val, bool):
                    set_nested_attr(env_cfg, toggle_attr, toggle_val)

    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "anymal_c_flat",
         "help": "Resume training or start testing from a checkpoint. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str, "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--run_name", "type": str, "help": "Name of the run. Overrides config file if provided."},
        {"name": "--load_run", "type": str,
         "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided."},
        {"name": "--checkpoint", "type": int,
         "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided."},

        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},

        {"name": "--feet_air_time", "type": float, "default": None, "help": "Time the feet are in the air during the trot gait"},
        {"name": "--hip_abduction_adduction", "type": float, "default": None,
         "help": "Leg spread weight for the reward shaping. Overrides config file if provided."},
        {"name": "--max_leg_spread", "type": float, "default": None,
         "help": "Maximum leg spread for the reward shaping. Overrides config file if provided."},
        {"name": "--foot_drag", "type": float, "default": None, "help": "Penalize foot drag for the reward shaping. Overrides config file if provided."},

        {"name": "--domain_rand_friction_range", "type": str, "default": None,
         "help": "Minimum and maximum values for the domain randomization friction parameter's range (format: 'min,max').", },
        {"name": "--domain_rand_ground_friction_range", "type": str, "default": None,
         "help": "Minimum and maximum values for the ground friction for the domain randomization.", },
        {"name": "--domain_rand_added_mass_range", "type": str, "default": None,
         "help": "Minimum and maximum values for the domain randomization added mass parameter's range (format: 'min,max').", },
        {"name": "--domain_rand_push_range", "type": str, "default": None,
         "help": "Minimum and maximum values for the domain randomization push parameter's range (format: 'min,max').", },
    ]
    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)

    # Parse range values from strings
    range_args = [
        "domain_rand_friction_range",
        "domain_rand_added_mass_range",
        "domain_rand_push_range",
        "domain_rand_ground_friction_range",
    ]

    # Convert string range values to float tuples
    for arg_name in range_args:
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            try:
                # Split by comma and convert to floats
                min_val, max_val = [float(x.strip()) for x in arg_value.split(',')]
                setattr(args, arg_name, [min_val, max_val])
            except (ValueError, AttributeError):
                print(f"Warning: Could not parse {arg_name}={arg_value}. Expected format: 'min,max'")
                setattr(args, arg_name, None)

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == 'cuda':
        args.sim_device += f":{args.sim_device_id}"
    return args


def set_nested_attr(obj, attr_path, value):
    """
    Set a nested attribute using dot notation.

    Args:
        obj: The object to set the attribute on
        attr_path: String with dot notation (e.g. "domain_rand.friction_range")
        value: The value to set
    """
    parts = attr_path.split('.')
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)

    # Debug print to verify setting worked
    print(f"Set {attr_path} to {value}, now value is {getattr(obj, parts[-1])}")


def export_policy_as_jit(actor_critic, path):
    if hasattr(actor_critic, 'memory_a'):
        # assumes LSTM: TODO add GRU
        exporter = PolicyExporterLSTM(actor_critic)
        exporter.export(path)
    else:
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy.pt')
        model = copy.deepcopy(actor_critic.actor).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.

    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


def export_policy_as_onnx(actor_critic, path, example_obs):
    """
    Exports a feed-forward actor to ONNX format.
    :param actor_critic: your PPO runner's actor_critic object
    :param path: directory where the .onnx file will be written
    :param example_obs: a torch.Tensor of observations (cpu) to trace the model
    """
    # detect recurrent policy
    if hasattr(actor_critic, 'memory_a'):
        raise NotImplementedError(
            "ONNX export for LSTM policies isn't implemented yet. "
            "You'd need to export both the model and its hidden/cell states."
        )
    # prepare target directory & model
    os.makedirs(path, exist_ok=True)
    deploy_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))  # "deploy_73feeca4"
    hash_suffix = deploy_folder.split('_', 1)[1]  # "73feeca4"
    onnx_path = os.path.join(path, f"policy_{hash_suffix}.onnx")
    # get a clean copy of the actor
    model = copy.deepcopy(actor_critic.actor).to('cpu')
    model.eval()
    # ensure batch dimension
    dummy_input = example_obs.detach().to('cpu')
    if dummy_input.dim() == 1:
        dummy_input = dummy_input.unsqueeze(0)
    # export
    torch.onnx.export(
        model.cpu().eval(),
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['obs'],
        output_names=['actions']
    )
    print(f'Exported policy as ONNX model to: {onnx_path}')
