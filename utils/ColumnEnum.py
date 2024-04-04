#from enum import StrEnum
from enum import Enum

class Columns(Enum):
    # categorical
    categorical_date = 'date'
    categorical_minute = 'minute'
    categorical_hour = 'hour'
    categorical_year = 'year'
    categorical_quarter = 'quarter'
    categorical_month = 'month'
    categorical_monthNumber = 'monthNumber'
    categorical_weekNumber = 'weekNumber'
    categorical_weekDay = 'weekDay'
    categorical_dateNumber = 'dateNumber'
    categorical_line = 'line'
    categorical_eventName = 'eventName'
    categorical_datetime = 'datetime'

    # weather
    weather_acc_precip = 'acc_precip'
    weather_bright_sunshine = 'bright_sunshine'
    weather_mean_cloud_cover = 'mean_cloud_cover'
    weather_mean_pressure = 'mean_pressure'
    weather_mean_relative_hum = 'mean_relative_hum'
    weather_mean_temp = 'mean_temp'
    weather_mean_wind_speed = 'mean_wind_speed'
    weather_snow_depth = 'snow_depth'

    # calendar
    calendar_peakHour = 'peakHour'
    calendar_Q1 = 'Q1'
    calendar_Q2 = 'Q2'
    calendar_Q3 = 'Q3'
    calendar_Q4 = 'Q4'
    calendar_mon = 'mon'
    calendar_tue = 'tue'
    calendar_wed = 'wed'
    calendar_thu = 'thu'
    calendar_fri = 'fri'
    calendar_sat = 'sat'
    calendar_sun = 'sun'
    calendar_workdayPlan = 'workdayPlan'
    calendar_saturdayPlan = 'saturdayPlan'
    calendar_sundayAndHolidayPlan = 'sundayAndHolidayPlan'
    calendar_summerVacation = 'summerVacation'
    calendar_fallVacation = 'fallVacation'
    calendar_christmasVacation = 'christmasVacation'
    calendar_winterVacation = 'winterVacation'
    calendar_easterVacation = 'easterVacation'
    calendar_event = 'event'

    # target
    target_passengersBoarding = 'passengersBoarding'
    target_diff = 'diff'

    # line
    line_1A = '1A'
    line_2A = '2A'
    line_3A = '3A'
    line_4A = '4A'
    line_5A = '5A'
    line_6A = '6A'

