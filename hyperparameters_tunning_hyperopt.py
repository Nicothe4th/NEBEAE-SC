from NEBEAE_SC import NEBEAESC
import vnirsynth as VNIR
import pandas as pd
from aux import errors1
from hyperopt import fmin, tpe, hp, Trials, STATUS_FAIL, STATUS_OK


# Objective function for Bayesian optimization
def objective(params):
    rho, nu, tau, lambdaTV = params['rho'], params['nu'], params['tau'], params['lambdaTV']

    
    parameters = [initcond, rho, lambdaTV, tau, nu, nRow, nCol, epsilon, maxiter, parallel, disp_iter]
    
    P, A, W, Ds, S, Yh, conv_track, sb_track = NEBEAESC(Yo, N, parameters)
    
    _,_,error_W_Ao = errors1(Yo, Yh, Ao, W, Po, P, N)
    
    return {'status': STATUS_OK, 'loss': error_W_Ao}

# Set up the data
nsamples = 100  # Size of the Squared Image nsamples x nsamples
noise = 40  # Level in dB of Noise 40, 35, 30, 25, 20
SNR = noise  
PSNR = noise  

Yo, Po, Ao, Do = VNIR.VNIRsynthNLM(nsamples, SNR, PSNR, 4)
N = 3

nCol = nRow = nsamples
epsilon = 1e-3
maxiter = 20
parallel = 0
disp_iter = 0
initcond = 6

# Define the search space
space = {
    'rho': hp.uniform('rho', 1e-3, .9),
    'nu': hp.uniform('nu', 1e-3, 1e5),
    'tau': hp.uniform('tau', 1e-5, 1e5),
    'lambdaTV': hp.uniform('lambdaTV', 1e-5, 1.0)
}

# Run the optimization
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)

# Collect results
results = []
for trial in trials.trials:
    if trial['result']['status'] == STATUS_OK:
        params = trial['misc']['vals']
        results.append({
            'rho': (params['rho'][0]),  # Convert back from log scale
            'nu': (params['nu'][0]),
            'tau': (params['tau'][0]),
            'lambdaTV': (params['lambdaTV'][0]),
            'loss': trial['result']['loss']
        })

# Create DataFrame
df = pd.DataFrame(results)
df = df.sort_values(by='loss')

best_rho = best['rho']
best_nu = best['nu']
best_tau = best['tau']
best_lambdaTV = best['lambdaTV']

# Create the final parameters vector
final_parameters = [initcond, best_rho, best_lambdaTV, best_tau, best_nu, nRow, nCol, epsilon, maxiter, parallel, disp_iter]

# Print best parameters and final parameters vector
print("Best parameters:")
print("rho: ", best_rho)
print("nu: ", best_nu)
print("tau: ", best_tau)
print("lambdaTV: ", best_lambdaTV)

print("Final parameters vector:")
print(final_parameters)

# Rerun the model with the best parameters
Pf, Af, Wf, Ds, S, Yhf, conv_track, sb_track = NEBEAESC(Yo, N, final_parameters)

# Compute the error_W_Ao with the best parameters
_,_,error_W_Ao =  errors1(Yo, Yhf, Ao, Wf, Po, Pf, N)
print("Final error_W_Ao: ", error_W_Ao)

# Verify consistency with the best loss reported during trials
best_trial_loss = df.iloc[0]['loss']
print("Best trial loss: ", best_trial_loss)
print("Difference: ", abs(best_trial_loss - error_W_Ao))

# Convert to LaTeX
latex_table = df.to_latex(index=False, float_format="%.3e")
print(latex_table)

# Save to a .tex file
with open("hyperopt_results.tex", "w") as f:
    f.write(latex_table)



