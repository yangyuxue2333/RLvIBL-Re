# Decoding Decision-Making Strategies Through Resting-State fMRI Data   

Two dominant framworks exists to explain how people make decisions from experience. In the _Reinforcement Learning_ (RL) frameowork, decisions are progressively calibrated by adjusting the expected value of an option throuh differences in reward predictions. In the _Instance Based Learning_ (IBL) and the _Decision by Sampling_ frameworks, on the other hand, decisions are supposedly made by sampling from memory the outcomes of previous choices.

The two approaches are very similar and make almost identical predictions, so much so that scientists often refer to them interchangeably. These two frameoworks, however entail different interpretations in the term of the neural circuitry driving the decision process.

In RL, the expected utilities of previous choices are stored are cached values associated with each option. These values do not decay over time, and the association between option and value is supposed to be learned through dopaminergic reinforcement of the basal ganglia circuitry. This suggests a procedural learning system.

In IBL, the expected utilities are stored as episodic memories, and decisions are made by either averaging over their values of estimating a mean by repeatedly sampling from memory. In this case, values are associated with each episodic trace, and the global value of an option is shaped by the previous history as well as the rules that shape memory forgetting. This framewro implies that decision-making is carried out by the hippocampal-prefrontal neural circuitry that underpins declarative memory encoding and retrieval.

Furthermore, it is possible that different individuals behave differently, and might rely on one circuit or another. If so, it would make sense that different idividuals play to their own specific strengths, and use the system that gives the best results, given their neural makeup.

## Distinguishing IBL from RL

Because IBL-based decisions relies on decaying traces of previous choices, it makes subtly different predictions than RL. To distinguish the two, we will model decision-making using an RL or IBL framework in ACT-R (an integrated cognitive architecture that allows to model both). Each model is fit to each individual, and the model whose parameters yield the best match (using Log Likelihood) will be taken as evidence of the decision-making strategy used.

## The Task

As a task, we are using the "Incentive Processing" task of the Human Connectome Project. Data is collected from N=176 participants, for whom both the behavioral data and resting-state fMRI is available. 

## Computational Modeling

### Code Location

* Source codes are located in `./script`

* Analysis codes are located in `./analysis`

* Data are in `./data`

### Run simulation (deprecated)

In this task, the order of trial stimuli has been fixed by each subject. So in order to run one simulation, 
you need to specify the subject number (i.e. `100307_fnca`). 

To run one round of simulation, go to `./script`

    from simulate import *

    simulate(epoch = 1, 
             subject_id = '100307_fnca', 
             param_id, 
             model='model1', 
             param_set={'ans':0.1}, 
             verbose=True, 
             log_dir=False, 
             special_suffix="")

### Run MaxLL analysis (deprecated)
To estimate best fit model, we use maxLL approach

Go to `/script`, run python script
    
    python -m run_maxLL.py
    

### Run parameter recovery analysis (deprecated)

Parameter recovery analysis: select a random set of parameter combination P, run 1 simulation for both model. 
Using the simulated output as fake subject data and feed into maxLL pipline to find the best fit parameter P'.
Examine the correlation between P and P'

    from simulate import *
    
    # load best fit parameter list (list of dict type)
    [{'subject_id': '994273_fnca',
                      'ans': 0.45,
                      'bll': 0.6,
                      'lf': 0.1,
                      'egs': 0.4,
                      'alpha': 0.25,
                      'r': 0.1,
                      'param_id': 'param0'}, ...]
    param_list = load_best_parameter_list(main_dir=main_dir, log_dir=subject_dir)
    
    # Start simulation
    run_maxLL_pr_pipline(main_dir=main_dir, subject_dir=subject_dir, param_list=param_list)


A simple way: Go to `/script`, run python script

Run one simulation for each participant using the best fit parameter set
 
    python -m run_pr_best

Run one simulation for each participant with random combination of parameter
    
- ans = [0.05, 0.2, 0.3, 0.4, 0.5]
- bll = [.2, .5, .8]
- lf = [0.1]
- egs =  [0.05, 0.2, 0.3, 0.4, 0.5]
- alpha = [.2, .5, .8]
- r = [0.1]
  

    python -m run_pr

---
## Data

rsfMRI preprocessed data are saved in data director

- `data/X.csv`: predictors
- `data/Y.csv`: class labels
- `data/LL_model2.csv`: LogLikelihood data from ACTR model simulation
- `data/power_2011.csv`: functional connectivity parcellation by Power et al. (2011)
