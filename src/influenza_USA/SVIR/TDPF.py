"""
This script contains the time-dependent parameter functions associated with the age-stratified spatially-explicit SIR model for use with pySODM.
"""

__author__      = "Tijs Alleman"
__copyright__   = "Copyright (c) 2024 by T.W. Alleman, IDD Group, Johns Hopkins Bloomberg School of Public Health. All Rights Reserved."

import numpy as np
from functools import lru_cache
from dateutil.easter import easter
from datetime import datetime, timedelta

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

class make_vaccination_function():

    def __init__(self, vaccination_data):
        """ Load the vaccination data and store them in class
        """
        # add week number for temporal indexing
        vaccination_data['week'] = vaccination_data['date'].dt.isocalendar().week
        vaccination_data = vaccination_data[['week', 'age', 'state', 'vaccination_rate']]
        self.vaccination_data = vaccination_data.groupby(by=['week', 'age', 'state']).last().sort_index().reset_index()    # make sure it's ordered
        # state sizes
        self.n_age = len(self.vaccination_data['age'].unique())
        self.n_loc = len(self.vaccination_data['state'].unique())
        pass

    @lru_cache() # avoid heavy IO while simulating
    def get_vaccination_rate(self, t):
        """ Returns the vaccination rates in a given week as an np.ndarray of shape (n_age, n_loc)
        """
        week_number = t.isocalendar().week
        try:
            return np.array(self.vaccination_data[self.vaccination_data['week'] == week_number]['vaccination_rate'].values, np.float64).reshape(self.n_age, self.n_loc) 
        except:
            return np.zeros([self.n_age, self.n_loc], np.float64)
    
    def vaccination_function(self, t, states, param, vaccine_rate_modifier, vaccine_rate_timdelta):
        """ pySODM compatible wrapper
        """
        return vaccine_rate_modifier * self.get_vaccination_rate(t + timedelta(days=vaccine_rate_timdelta))

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
        return self.__call__(t)

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