"""
This script contains the time-dependent parameter functions associated with the age-stratified spatially-explicit SIR model for use with pySODM.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import numpy as np
import tensorflow as tf
from functools import lru_cache
from dateutil.easter import easter
from datetime import datetime, timedelta

##################################
## Hierarchal transmission rate ##
##################################

class hierarchal_transmission_rate_function():

    def __init__(self):

        self.region_mapping = np.array([5,  # Alabama (01000) - East South Central
                                        8,  # Alaska (02000) - Pacific
                                        7,  # Arizona (04000) - Mountain
                                        6,  # Arkansas (05000) - West South Central
                                        8,  # California (06000) - Pacific
                                        7,  # Colorado (08000) - Mountain
                                        0,  # Connecticut (09000) - New England
                                        4,  # Delaware (10000) - South Atlantic
                                        4,  # District of Columbia (11000) - South Atlantic
                                        4,  # Florida (12000) - South Atlantic
                                        4,  # Georgia (13000) - South Atlantic
                                        8,  # Hawaii (15000) - Pacific
                                        7,  # Idaho (16000) - Mountain
                                        2,  # Illinois (17000) - East North Central
                                        2,  # Indiana (18000) - East North Central
                                        3,  # Iowa (19000) - West North Central
                                        3,  # Kansas (20000) - West North Central
                                        5,  # Kentucky (21000) - East South Central
                                        6,  # Louisiana (22000) - West South Central
                                        0,  # Maine (23000) - New England
                                        4,  # Maryland (24000) - South Atlantic
                                        0,  # Massachusetts (25000) - New England
                                        2,  # Michigan (26000) - East North Central
                                        3,  # Minnesota (27000) - West North Central
                                        5,  # Mississippi (28000) - East South Central
                                        3,  # Missouri (29000) - West North Central
                                        7,  # Montana (30000) - Mountain
                                        3,  # Nebraska (31000) - West North Central
                                        7,  # Nevada (32000) - Mountain
                                        0,  # New Hampshire (33000) - New England
                                        1,  # New Jersey (34000) - Mid-Atlantic
                                        7,  # New Mexico (35000) - Mountain
                                        1,  # New York (36000) - Mid-Atlantic
                                        4,  # North Carolina (37000) - South Atlantic
                                        3,  # North Dakota (38000) - West North Central
                                        2,  # Ohio (39000) - East North Central
                                        6,  # Oklahoma (40000) - West South Central
                                        8,  # Oregon (41000) - Pacific
                                        1,  # Pennsylvania (42000) - Mid-Atlantic
                                        0,  # Rhode Island (44000) - New England
                                        4,  # South Carolina (45000) - South Atlantic
                                        3,  # South Dakota (46000) - West North Central
                                        5,  # Tennessee (47000) - East South Central
                                        6,  # Texas (48000) - West South Central
                                        7,  # Utah (49000) - Mountain
                                        0,  # Vermont (50000) - New England
                                        4,  # Virginia (51000) - South Atlantic
                                        8,  # Washington (53000) - Pacific
                                        4,  # West Virginia (54000) - South Atlantic
                                        2,  # Wisconsin (55000) - East North Central
                                        7,  # Wyoming (56000) - Mountain
                                        8,  # Puerto Rico (72000) - Assumed Pacific
                                        ])

    def __call__(self, t, states, param, beta_US, delta_beta_spatial, delta_beta_temporal,
                 delta_beta_spatial_Nov, delta_beta_spatial_Dec, delta_beta_spatial_Jan, delta_beta_spatial_Feb, delta_beta_spatial_Mar):
        """
        A function constructing a spatio-temporal hierarchal transmission rate 'beta'

        input
        -----

        t: datetime.datetime
            current timestep in simulation

        states: dict
            current values of all model states
        
        param: dict
            current values of all model parameters

        beta_US: float
            overall transmission rate. hierarchal level 0.

        delta_beta_spatial: np.ndarray (len: 9)
            a spatial modifier on the overall transmision rate for every US region. hierarchal level 1.

        delta_beta_temporal: np.ndarray (len: 4)
            a temporal modifier on the overall transmission rate for Dec, Jan, Feb, Mar. hierarchal level 1.

        delta_beta_spatial_Nov: np.ndarray (len: 9)
            a spatio-temporal modifier for every US region in Nov. hierarchal level 2.

        output
        ------

        beta: np.ndarray
            transmission rate per US state
        """
        # region to states
        delta_beta_spatial = delta_beta_spatial[self.region_mapping]
        delta_beta_spatial_Nov = delta_beta_spatial_Nov[self.region_mapping]
        delta_beta_spatial_Dec = delta_beta_spatial_Dec[self.region_mapping]
        delta_beta_spatial_Jan = delta_beta_spatial_Jan[self.region_mapping]
        delta_beta_spatial_Feb = delta_beta_spatial_Feb[self.region_mapping]
        delta_beta_spatial_Mar = delta_beta_spatial_Mar[self.region_mapping]

        # construct beta
        if t.month == 11:
            beta = beta_US * (delta_beta_temporal[0] + 1) * (delta_beta_spatial + 1)  * (delta_beta_spatial_Nov + 1)
        elif t.month == 12:
            beta = beta_US * (delta_beta_temporal[1] + 1) * (delta_beta_spatial + 1)  * (delta_beta_spatial_Dec + 1)
        elif t.month == 1:
            beta = beta_US * (delta_beta_temporal[2] + 1) * (delta_beta_spatial + 1)  * (delta_beta_spatial_Jan + 1)
        elif t.month == 2:
            beta = beta_US * (delta_beta_temporal[3] + 1) * (delta_beta_spatial + 1)  * (delta_beta_spatial_Feb + 1)
        elif t.month == 3:
            beta = beta_US * (delta_beta_temporal[4] + 1) * (delta_beta_spatial + 1)  * (delta_beta_spatial_Mar + 1)
        else:
            beta = beta_US * (delta_beta_spatial + 1) 

        return beta
    
##############
## Vaccines ##
##############

class make_vaccination_function():

    def __init__(self, season, vaccination_data):
        """ Format the vaccination data
        """

        # check input season
        if ((season not in vaccination_data.index.unique().values) & (season != 'average')):
            raise ValueError(f"season '{season}' vaccination data not found. provide a valid season (format '20xx-20xx') or 'average'.")

        # drop index
        vaccination_data = vaccination_data.reset_index()

        if season != 'average':
            # slice out correct season
            vaccination_data = vaccination_data[vaccination_data['season'] == season]
            # add week number & remove date
            vaccination_data['week'] = vaccination_data['date'].dt.isocalendar().week.values
            vaccination_data = vaccination_data[['week', 'age', 'state', 'daily_incidence']]
            # sort age groups / spatial units --> are sorted in the model
            vaccination_data = vaccination_data.groupby(by=['week', 'age', 'state']).last().sort_index().reset_index()
            # remove negative entries (there may be some in the first week(s) of the season)
            vaccination_data['daily_incidence'] = np.where(vaccination_data['daily_incidence'] < 0, 0, vaccination_data['daily_incidence'])
        else:
            # add week number & remove date
            vaccination_data['week'] = vaccination_data['date'].dt.isocalendar().week.values
            vaccination_data = vaccination_data[['week', 'age', 'state', 'daily_incidence']]
            # average out + sort
            vaccination_data = vaccination_data.groupby(by=['week', 'age', 'state']).mean('daily_incidence').sort_index().reset_index()
            # remove negative entries (there may be some in the first week(s) of the season)
            vaccination_data['daily_incidence'] = np.where(vaccination_data['daily_incidence'] < 0, 0, vaccination_data['daily_incidence'])

        # assign to object
        self.vaccination_data = vaccination_data

        # compute state sizes
        self.n_age = len(self.vaccination_data['age'].unique())
        self.n_loc = len(self.vaccination_data['state'].unique())

    @lru_cache() # avoid heavy IO while simulating
    def get_vaccination_incidence(self, t):
        """ Returns the daily vaccination incidence as an np.ndarray of shape (n_age, n_loc)
        """
        week_number = t.isocalendar().week
        try:
            return np.array(self.vaccination_data[self.vaccination_data['week'] == week_number]['daily_incidence'].values, np.float64).reshape(self.n_age, self.n_loc) 
        except:
            return np.zeros([self.n_age, self.n_loc], np.float64)
    
    def vaccination_function(self, t, states, param, vaccine_incidence_modifier, vaccine_incidence_timedelta):
        """ pySODM compatible wrapper
        """
        return vaccine_incidence_modifier * self.get_vaccination_incidence(t - timedelta(days=vaccine_incidence_timedelta))

#####################
## Social contacts ##
#####################

class make_contact_function():

    def __init__(self, contact_matrix_week_noholiday, contact_matrix_week_holiday, contact_matrix_weekend):
        """ Load the contact matrices and stores them in class
        """
        self.contact_matrix_week_noholiday = contact_matrix_week_noholiday
        self.contact_matrix_week_holiday = contact_matrix_week_holiday
        self.contact_matrix_weekend = contact_matrix_weekend

        pass

    @lru_cache()
    def __call__(self, t):
        """ Returns the right contact matrix depending on daytype; cached to avoid heavy IO referencing during simulation
        """
        if t.weekday() >= 5:
            return self.contact_matrix_weekend
        else:
            if self.is_school_holiday(t):
                return self.contact_matrix_week_holiday
            else:
                return self.contact_matrix_week_noholiday
            
    def contact_function(self, t, states, param):
        """ pySODM compatible wrapper
        """
        return tf.convert_to_tensor(self.__call__(t), dtype=float)

    @staticmethod
    def is_school_holiday(d):
        """
        A function returning 'True' if a given date is a school holiday for primary and secundary schools in the USA.
        Tertiary education is not considered in this work.

        Input
        =====

        d: datetime.datetime
            current date

        Returns
        =======

        is_school_holiday: bool
            True: date `d` is a school holiday for primary and secundary schools
        """

        #########################
        ## Fixed date holidays ##
        #########################

        # summer vacation
        if d.month in [6, 7, 8]:  
            return True

        # fixed-date holidays
        if (d.month, d.day) in {(1, 1): "New Year's Day", (7, 4): "Independence Day", (11, 11): "Veterans Day", (12, 25): "Christmas", (6,19): "Juneteenth"}:
            return True

        #############################
        ## winter and spring break ##
        #############################

        holiday_weeks = []
        # Winter break
        # Typically two week break covering Christmas and NY --> Similar to Belgium
        # If Christmas falls on Saturday or Sunday, winter break starts week after
        w_christmas_current = datetime(
            year=d.year, month=12, day=25).isocalendar().week
        if datetime(year=d.year, month=12, day=25).isoweekday() in [6, 7]:
            w_christmas_current += 1
        w_christmas_previous = datetime(
            year=d.year-1, month=12, day=25).isocalendar().week
        if datetime(year=d.year-1, month=12, day=25).isoweekday() in [6, 7]:
            w_christmas_previous += 1
        # Christmas "logic"
        if w_christmas_previous == 52:
            if datetime(year=d.year-1, month=12, day=31).isocalendar().week != 53:
                holiday_weeks.append(1)
        if w_christmas_current == 51:
            holiday_weeks.append(w_christmas_current)
            holiday_weeks.append(w_christmas_current+1)
        if w_christmas_current == 52:
            holiday_weeks.append(w_christmas_current)
            if datetime(year=d.year, month=12, day=31).isocalendar().week == 53:
                holiday_weeks.append(w_christmas_current+1)
        holiday_weeks = [1,] # two weeks might be a huge overestimation of the impact of this holiday --> first week of year is most consistent with data

        # Spring break
        # Extract date of easter
        d_easter = easter(d.year)
        # Convert from datetime.date to datetime.datetime
        d_easter = datetime(d_easter.year, d_easter.month, d_easter.day)
        # Get week of easter
        w_easter = d_easter.isocalendar().week
        # Default logic: Easter holiday starts first monday of April
        # Unless: Easter falls after 04-15: Easter holiday ends with Easter
        # Unless: Easter falls in March: Easter holiday starts with Easter
        if d_easter >= datetime(year=d.year, month=4, day=15):
            w_easter_holiday = w_easter - 1
        elif d_easter.month == 3:
            w_easter_holiday = w_easter + 1
        else:
            w_easter_holiday = datetime(
                d.year, 4, (8 - datetime(d.year, 4, 1).weekday()) % 7).isocalendar().week
        holiday_weeks.append(w_easter_holiday)
        holiday_weeks.append(w_easter_holiday+1)

        # Check winter and spring break
        if d.isocalendar().week in holiday_weeks:
            return True

        #######################
        ## Variable holidays ##
        #######################

        ## MLKJ day: Third monday of January
        jan_first = datetime(d.year, 1, 1)                                              # start with january first
        first_monday = jan_first + timedelta(days=(7 - jan_first.weekday()) % 7)        # find first monday
        date = first_monday + timedelta(weeks=2)                                        # add to weeks to it
        if d == date:
            return True

        ## Presidents day: Third monday of February
        feb_first = datetime(d.year, 2, 1)  
        first_monday = feb_first + timedelta(days=(7 - feb_first.weekday()) % 7)
        date = first_monday + timedelta(weeks=2)
        if d == date:
            return True

        ## Memorial day: Last monday of May
        may_first = datetime(d.year, 5, 1)  
        first_monday = may_first + timedelta(days=(7 - may_first.weekday()) % 7)
        date = first_monday + timedelta(weeks=3)
        if d == date:
            return True

        ## Labor day: First monday of September
        sept_first = datetime(d.year, 9, 1)                                 
        first_monday = sept_first + timedelta(days=(7 - sept_first.weekday()) % 7)
        date = first_monday + timedelta(weeks=0)
        if d == date:
            return True

        ## Columbus day: Second monday of October
        oct_first = datetime(d.year, 10, 1)                                 
        first_monday = oct_first + timedelta(days=(7 - oct_first.weekday()) % 7)
        date = first_monday + timedelta(weeks=0)
        if d == date:
            return True

        ## Thanksgiving day: Fourth Thursday of November
        nov_first = datetime(d.year, 11, 1)  
        first_thursday = nov_first + timedelta(days=(3 - nov_first.weekday() + 7) % 7)
        date = first_thursday + timedelta(weeks=3)
        if d == date:
            return True
        
        return False

################################
## Initial condition function ##
################################

from influenza_USA.SVIR.utils import construct_initial_susceptible, get_cumulative_vaccinated
class make_initial_condition_function():

    def __init__(self, spatial_resolution, age_resolution, start_sim, season, vaccination_data):
        # retrieve the demography (susceptible pool)
        self.demography = construct_initial_susceptible(spatial_resolution, age_resolution)
        # retrieve the cumulative vaccinated individuals at `start_sim` in `season`
        self.vaccinated = get_cumulative_vaccinated(start_sim, season, vaccination_data)
        self.region_mapping = np.array([
            5,  # Alabama (01000) - East South Central
            8,  # Alaska (02000) - Pacific
            7,  # Arizona (04000) - Mountain
            6,  # Arkansas (05000) - West South Central
            8,  # California (06000) - Pacific
            7,  # Colorado (08000) - Mountain
            0,  # Connecticut (09000) - New England
            4,  # Delaware (10000) - South Atlantic
            4,  # District of Columbia (11000) - South Atlantic
            4,  # Florida (12000) - South Atlantic
            4,  # Georgia (13000) - South Atlantic
            8,  # Hawaii (15000) - Pacific
            7,  # Idaho (16000) - Mountain
            2,  # Illinois (17000) - East North Central
            2,  # Indiana (18000) - East North Central
            3,  # Iowa (19000) - West North Central
            3,  # Kansas (20000) - West North Central
            5,  # Kentucky (21000) - East South Central
            6,  # Louisiana (22000) - West South Central
            0,  # Maine (23000) - New England
            4,  # Maryland (24000) - South Atlantic
            0,  # Massachusetts (25000) - New England
            2,  # Michigan (26000) - East North Central
            3,  # Minnesota (27000) - West North Central
            5,  # Mississippi (28000) - East South Central
            3,  # Missouri (29000) - West North Central
            7,  # Montana (30000) - Mountain
            3,  # Nebraska (31000) - West North Central
            7,  # Nevada (32000) - Mountain
            0,  # New Hampshire (33000) - New England
            1,  # New Jersey (34000) - Mid-Atlantic
            7,  # New Mexico (35000) - Mountain
            1,  # New York (36000) - Mid-Atlantic
            4,  # North Carolina (37000) - South Atlantic
            3,  # North Dakota (38000) - West North Central
            2,  # Ohio (39000) - East North Central
            6,  # Oklahoma (40000) - West South Central
            8,  # Oregon (41000) - Pacific
            1,  # Pennsylvania (42000) - Mid-Atlantic
            0,  # Rhode Island (44000) - New England
            4,  # South Carolina (45000) - South Atlantic
            3,  # South Dakota (46000) - West North Central
            5,  # Tennessee (47000) - East South Central
            6,  # Texas (48000) - West South Central
            7,  # Utah (49000) - Mountain
            0,  # Vermont (50000) - New England
            4,  # Virginia (51000) - South Atlantic
            8,  # Washington (53000) - Pacific
            4,  # West Virginia (54000) - South Atlantic
            2,  # Wisconsin (55000) - East North Central
            7,  # Wyoming (56000) - Mountain
            8,  # Puerto Rico (72000) - Assumed Pacific
            ])
        
    def initial_condition_function(self, f_I, f_R, delta_f_R):
        """
        A function setting the model's initial condition. Uses a hierarchal structure for the initial immunity.
        
        input
        -----

        f_I: float / np.ndarray
            fraction of the unvaccinated population infected at simulation start

        f_R: float
            initial immunity of US (parent distribution)
        
        delta_f_R: np.ndarray (len: 9)
            initial immunity modifier of US regions (child distributions)

        output
        ------

        initial_condition: dict
            Keys: 'S', 'V', 'I', 'R'. Values: np.ndarray (n_age x n_loc).
        """

        # convert delta_f_R from region to state level
        delta_f_R = delta_f_R[self.region_mapping]

        # construct hierarchal initial immunity
        f_R = f_R * (1 + delta_f_R) 

        return {'S':  self.demography - (1 - f_I - f_R) * self.vaccinated - (f_I + f_R) * self.demography,
                'V': (1 - f_I - f_R) * self.vaccinated,
                'I': f_I * self.demography,
                'R': f_R * self.demography}

################################################################
## PARKING: seasonality & exponential vaccine efficacy waning ##
################################################################

def seasonality_function(t, states, param, amplitude, peak_shift):
    """
    Default output function. Returns a sinusoid with average value 1.
    
    t : datetime.datetime
        simulation time
    amplitude : float
        maximum deviation of output with respect to the average (1)
    peak_shift : float
        phase. Number of days after January 1st after which the maximum value of the seasonality rescaling is reached 
    """
    ref_date = datetime(2021,1,1)
    # If peak_shift = 0, the max is on the first of January
    maxdate = ref_date + timedelta(days=float(peak_shift))
    # One period is one year long (seasonality)
    t = (t - maxdate)/timedelta(days=1)/365
    rescaling = 1 + amplitude*np.cos( 2*np.pi*(t))
    return rescaling

def exponential_waning_function(t, states, param, waning_start, T_s):
    if t < waning_start:
        return 1.0
    else:
        return np.exp(-1/T_s * float((t - waning_start)/timedelta(days=1)))
