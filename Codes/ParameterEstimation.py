# Here it is assumed a time varying perceived risk of vaccine side effects,
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy import integrate, optimize, interpolate
import time
import numpy.random as random
import multiprocessing
from multiprocessing import Pool
import sys
import os

#  #################################Saving Directory###########################################
commonPath = 'CAN/'
inp = sys.argv
dataFile = commonPath + 'EpiVacFile_6_1_week.csv'
MandateFile = commonPath + 'mandate_date.csv'
SEff = commonPath + 'SideEffect_Nov23.csv'
#  #################################Reading Directory#########################################
sideEffect = pd.read_csv(SEff)
OWID = pd.read_csv(dataFile).reset_index().assign(Date=lambda x: pd.to_datetime(x['Date'])).sort_values('Date')
method0 = 'LSODA'
StateName = list(OWID['Location'].unique())
mandate = pd.read_csv(MandateFile).assign(announcement=lambda x: pd.to_datetime(x['announcement']))
iter_interval = 2000
script_name = os.path.basename(__file__)

def system_dynamics(x, y, alpha, cv1, cbar, c_i, k1, *arg_N):
    #unpacking the arguments
    leng = 5
    lenn, N, fr_Ne, Ntt, man_date = arg_N[0:leng]
    time_ref = arg_N[leng:leng + lenn]
    number_of_case = arg_N[leng + lenn: leng + 2 * lenn]
    number_of_death = arg_N[leng + 2 * lenn: leng + 3 * lenn]
    frac_avail_dose = arg_N[leng + 3 * lenn: leng + 4 * lenn]
    side_effect_trend = arg_N[leng + 4 * lenn: leng + 4 * lenn + 5]
    side_effect_date = arg_N[leng + 4 * lenn + 5: leng + 4 * lenn + 10]
    #Interpolating
    number_of_case_interpolated = interpolate.interp1d(time_ref, number_of_case, kind='zero', axis=0, fill_value=
    "extrapolate")
    number_of_death_interpolated = interpolate.interp1d(time_ref, number_of_death, kind='zero', axis=0, fill_value=
    "extrapolate")
    frac_avail_dose_interpolated = interpolate.interp1d(time_ref, frac_avail_dose, kind='zero', axis=0, fill_value=
    "extrapolate")
    side_effect_interpolated = interpolate.interp1d(side_effect_date, side_effect_trend, kind='zero', axis=0, fill_value=
    (side_effect_trend[0],side_effect_trend[-1]), bounds_error=False)
    im_vax, br_vax, = y
    totalvax = im_vax + br_vax  #proportion of vaccinated
    frac_avail_dose_interpolated = frac_avail_dose_interpolated(x) - totalvax  #total available doses at time x /N
    if frac_avail_dose_interpolated < 0.0:
        frac_avail_dose_interpolated = 0.0
    cv = cv1*side_effect_interpolated(x) # put cv equal to 0 and comment this if the perceived vaccine side effect is constant
    excess_payoff = -cv + cbar + c_i * (number_of_case_interpolated(x) / Ntt) +\
                    (number_of_death_interpolated(x) / Ntt)
    # Compute intermediate values
    intermediate_im = (1 - alpha) * fr_Ne - im_vax
    intermediate_br = alpha * fr_Ne - br_vax
    # Calculate num_im_reg_m, num_im_reg_w, num_br_reg_m, and num_br_reg_w using the conditions
    conditions = {
        'num_im_reg': intermediate_im * totalvax *  excess_payoff if excess_payoff > 0.0 and
                                                                                intermediate_im > 0 else 0.0,
        'num_br_reg': intermediate_br if excess_payoff > 0.0 and intermediate_br > 0 else 0.0
    }
    # Calculate total_reg
    total_reg = sum(conditions.values())

    # Checking if enough doses are available
    if total_reg > frac_avail_dose_interpolated:
        factor = frac_avail_dose_interpolated * k1 / total_reg
    else:
        factor = k1
    dot_num_im_vax = factor * conditions['num_im_reg']
    dot_num_br_vax = factor * conditions['num_br_reg']

    return dot_num_im_vax, dot_num_br_vax

#OBJ_ODE performs the optimization
def OBJ_ODE(parameters, *arg):
    number_of_elements = 5
    ssize, N, fr_Ne, Ntot,  mandate_date = arg[0:number_of_elements]
    arg_N = [ssize, N, fr_Ne, Ntot, mandate_date]
    Time_Ref = arg[number_of_elements: number_of_elements + ssize]
    new_vaccinated = arg[number_of_elements + ssize:number_of_elements + 2 * ssize]
    Number_of_Case = arg[number_of_elements + 2 * ssize:number_of_elements + 3 * ssize]
    Number_of_Death = arg[number_of_elements + 3 * ssize:number_of_elements + 4 * ssize]
    Fraction_of_Eligible = arg[number_of_elements + 4 * ssize:number_of_elements + 5 * ssize]
    Side_effect_trend = arg[number_of_elements + 5 * ssize : number_of_elements + 5 * ssize + 5]
    Side_effect_date = arg[number_of_elements + 5 * ssize + 5:]
    # unpacking the parameters
    alpha_1_1, cv1,  cv_bar01, tilde_c_i_1,  k1 = parameters
    #solving the ODE
    sol = integrate.solve_ivp(system_dynamics, [Time_Ref[0], Time_Ref[-1]], (
                            0, 0), args=(alpha_1_1, cv1, cv_bar01, tilde_c_i_1,
                                                                             k1,   *arg_N, *Time_Ref,
                                                                             *Number_of_Case, *Number_of_Death,
                                                                             *Fraction_of_Eligible, *Side_effect_trend,
                                         *Side_effect_date),
                          t_eval=Time_Ref, dense_output=True, method=method0)
    #Calculate the model output
    model_output = N* sol.y[0, :] + N*sol.y[1, :]
    #Shift the model output
    model_output_shifted = np.insert(model_output, 0, 0)
    model_output_shifted = np.delete(model_output_shifted, [-1])
    #Calculate the RSE
    residuals_22 = np.asarray(model_output - model_output_shifted) - np.asarray(new_vaccinated)
    RSE_22 = np.sum(residuals_22 ** 2)
    return RSE_22


def optm(arg):
    i, seed_initial = arg
    Ref_Time = []
    total_vaccinated = (OWID.loc[OWID['Location'] == i, 'Admin_Dose_1']).to_numpy(dtype=np.float64)
    # Compute new vaccinated individuals
    total_shifted = np.insert(total_vaccinated, 0, 0)[:-1]
    new_vaccinated = total_vaccinated - total_shifted
    Number_of_Case = OWID.loc[OWID['Location'] == i, 'new_case'].to_numpy(dtype=np.float64)
    Number_of_Death = OWID.loc[OWID['Location'] == i, 'new_death'].to_numpy(dtype=np.float64)
    no0_ind = Number_of_Case != 0
    death_case_fraction = Number_of_Death[no0_ind]/  Number_of_Case[no0_ind]
    min_death_case = np.min(death_case_fraction)
    if min_death_case > 0.0:
        max_ci_scaled = min_death_case
    elif i=='PE': #number of death for PE was zero
        max_ci_scaled = 0.01
    else:
        non_min_val = death_case_fraction[death_case_fraction != min_death_case]
        max_ci_scaled = np.min(non_min_val)
    total_doses = OWID.loc[OWID['Location'] == i, 'dose_distributed'].to_numpy(dtype=np.float64)
    N = OWID.loc[OWID['Location'] == i, 'popf12'].iloc[0]
    fr_Ne = OWID.loc[OWID['Location'] == i, 'Ne'].iloc[0] / N
    Ntotal = OWID.loc[OWID['Location'] == i, 'poptotal'].iloc[0]
    # Extract mandate date
    announcement_date = mandate.loc[mandate['Location'] == 'CAN_' + i, 'announcement'].iloc[0]
    ManDate = pd.to_datetime(announcement_date)
    fraction_doses = np.clip(total_doses/N, 0, None)
    Ref_Time_2 = list(OWID.loc[OWID['Location'] == i, 'Date'])
    # Compute mandate indicator
    indicator = [1 if date > ManDate else 0 for date in Ref_Time_2]
    for k in Ref_Time_2:
        delta = k - Ref_Time_2[0]
        Ref_Time.append(delta.days)
    Time_Ref = np.array(Ref_Time, dtype=np.float64)
    Time_Ref = np.divide(Time_Ref, 7)
    mandateDate = np.argmax(np.asarray(indicator))
    # Determine side ID for regional data
    region_mapping = {
        'NB': 'ATL', 'NS': 'ATL', 'NL': 'ATL', 'PE': 'ATL',
        'MB': 'MB', 'SK': 'MB', 'NU': 'MB',
        'BC': 'BC', 'YT': 'BC',
        'AB': 'AB', 'NT': 'AB'
    }
    sideID = region_mapping.get(i, i)
    # d2 contains the average date of each impact Canada's wave ranging from Wave 10 till Wave 14
    d2 = np.array([pd.to_datetime('13/12/2020', format='%d/%m/%Y'), pd.to_datetime('13/02/2021', format='%d/%m/%Y'),
                   pd.to_datetime('20/03/2021', format='%d/%m/%Y'),
                   pd.to_datetime('08/05/2021', format='%d/%m/%Y'), pd.to_datetime('26/06/2021', format='%d/%m/%Y')])
    # Initialize an array to store the indices
    nearest_indices = np.zeros(len(d2), dtype=int)
    # Find the nearest index of Time_Ref for each element in d2
    nearest_indice = np.abs(np.subtract.outer(Ref_Time_2, d2)).argmin(axis=0)
    # Find the nearest index of Time_Ref for each element in d2
    nearest_indices = [Time_Ref[idx] for idx in nearest_indice]
    trend_sideEffect = sideEffect.loc[sideEffect['region'] == sideID, 'concerned_percentage'].to_numpy(dtype=np.float64)
    max_alpha, min_alpha, max_cv, min_cv, max_cbar0, min_cbar0 = 1.0, 0.0, 1.0, 0.0, 1.0, 0.0
    max_ci, min_ci, max_cd, min_cd, max_rate, min_rate = max_ci_scaled, 0.0, 1.0, 0.0, 10.0, 0.0
    np.random.seed(seed_initial)
    cv_g = random.random()
    cv_bar0_g = max_cbar0*random.random()
    tilde_C_I_g = 0 + max_ci * random.random()
    rate_g = 0 + max_rate * random.random()
    alpha_g = 0 + random.random()
    seed0 = seed_initial
    # if the perceived vaccine side effect is constant, then cv_g is removed and min_cbar0 is set to -1
    int_guess = np.array([alpha_g, cv_g,  cv_bar0_g,  tilde_C_I_g,  rate_g])
    limit_list = [(min_alpha,max_alpha), (min_cv,max_cv), (min_cbar0, max_cbar0),
                  (min_ci, max_ci), (min_rate,max_rate)]
    startTime = time.time()
    res_opt = optimize.dual_annealing(OBJ_ODE, bounds=limit_list, x0=int_guess,  maxiter=
        iter_interval, seed=random.default_rng(seed=seed0), initial_temp=50000, args=[len(Time_Ref), N, fr_Ne, Ntotal,
                                                                                       mandateDate,
                                                                                      *Time_Ref, *new_vaccinated,
                                                              *Number_of_Case, *Number_of_Death,
                                                              *fraction_doses, *trend_sideEffect, *nearest_indices])

    alpha_1, cv,  cv_bar0, tilde_c_i,  rate = res_opt.x
    RSE_1 = res_opt.fun

    with open(commonPath +  str(iter_interval) +  str(script_name)
              +  '.txt', "a") as text_file:
            print(f"{i}, alpha_1: {alpha_1}, cv: {cv}, cv_bar0: {cv_bar0},  "
                  f"c_i: {tilde_c_i},  "
                  f"rate: {rate}"
                  f" RSE: {RSE_1}, seed:{seed_initial}, ElapsedTime: {time.time()-startTime}", file=text_file)

    return i, alpha_1, cv,  cv_bar0, tilde_c_i,  rate,  RSE_1, seed_initial


def main():
    num_runs = 5
    result_dataframe = pd.DataFrame( columns=['Location', 'alpha_1', 'cv',  'cvbar0',  'c_i',
                                    'rate', 'RSE_1','seed'])
    initial_seed = 2024
    Number_of_cpus = multiprocessing.cpu_count()
    for province in StateName:
        seed_list = [initial_seed + run for run in range(num_runs)]
        Provinces = [province]*len(seed_list)
        with Pool(processes=Number_of_cpus) as run_pool:
            parallel_output = run_pool.map(optm, zip(Provinces, seed_list))
            run_pool.close()
            run_pool.join()
        ddf = pd.DataFrame(parallel_output, columns=['Location', 'alpha_1', 'cv',  'cvbar0',  'c_i',
                                    'rate', 'RSE_1', 'seed'])
        result_dataframe = pd.concat([ddf,result_dataframe])
    filepath = Path(commonPath + str(iter_interval) +
                    str(script_name))
    filepath.parent.mkdir(parents=True, exist_ok=True)
    result_dataframe.to_csv(filepath)


if __name__ == '__main__':
    main()

