import math
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


def Outputs(data):
    return 1.0 / (1.0 + np.exp(-data))


def GPIndividual1(data):
    predictions = np.tanh((3.33333 - data['action_type_Jump Bank Shot']) *
        (np.minimum(data['shot_zone_area_Center(C)'], -data[
        'action_type_Jump Bank Shot']) + (data['lat'] - np.exp(-(np.maximum
        (data['shot_zone_area_Center(C)'], np.exp(-(data[
        'shot_zone_area_Center(C)'] * 2.0 * (data[
        'shot_zone_area_Center(C)'] * 2.0))) / 2.0) * np.maximum(data[
        'shot_zone_area_Center(C)'], np.exp(-(data[
        'shot_zone_area_Center(C)'] * 2.0 * (data[
        'shot_zone_area_Center(C)'] * 2.0))) / 2.0))) * 2.0) - data[
        'distance_5'])) + np.tanh(np.maximum(data[
        'action_type_Step Back Jump shot'], np.tanh(data[
        'action_type_Fadeaway Bank shot'] - np.maximum(data[
        'action_type_Running Jump Shot'], data[
        'shot_zone_basic_Restricted Area'] * 2.0)) * (data[
        'action_type_Pullup Jump shot'] - np.tanh(np.maximum(0.472727, np.
        exp(-(data['distance_25'] * data['distance_25']))))) * 2.0) * 42.5
        ) + np.tanh(6.0625 * -(data['action_type_Jump Shot'] + (0.470588 +
        data['action_type_Tip Shot']) / 2.0)) + np.tanh((data[
        'action_type_Running Jump Shot'] - (data['action_type_Layup Shot'] -
        data['shot_zone_range_Less Than 8 ft.'] + np.tanh(data[
        'action_type_Fadeaway Bank shot'] / 2.0) * 2.0) + np.tanh(data[
        'action_type_Fadeaway Jump Shot'])) * 2.0 * 2.0) + np.tanh((data[
        'shot_zone_area_Right Side Center(RC)'] + data['opponent_MIA']) / 
        2.0 - (data['action_type_Jump Shot'] + data['action_type_Jump Shot'
        ] * ((6.0625 + (3.141593 - data['season_2000-01'] / 2.0 * 2.0 * 2.0
        )) * 2.0)) / 2.0) + np.tanh((0.147727 + (data[
        'action_type_Turnaround Bank shot'] - -((data[
        'action_type_Slam Dunk Shot'] + ((data[
        'action_type_Driving Dunk Shot'] + data[
        'action_type_Slam Dunk Shot']) / 2.0 - data['distance_45'])) / 2.0 -
        np.maximum((data['shot_zone_range_16-24 ft.'] <= (0.490566 + data[
        'opponent_ATL']) / 2.0).astype(float), -(data[
        'action_type_Finger Roll Shot'] / 2.0)))) * 2.0) / 2.0) + np.tanh(
        data['shot_zone_range_8-16 ft.'] + (np.maximum(data[
        'shot_zone_basic_Mid-Range'], data['distance_0']) - -data[
        'action_type_Bank Shot']) + (data['distance_1'] + data['distance_0'])
        ) + np.tanh(np.maximum((data['action_type_Jump Bank Shot'] + data[
        'action_type_Driving Dunk Shot']) / 2.0 / 2.0, data[
        'action_type_Running Bank shot']) - (data['period_4'] - (data[
        'shot_zone_area_Right Side Center(RC)'] + (0.0 - (data[
        'last_moments_1'] - np.minimum(data['action_type_Running Hook Shot'
        ], np.tanh((data['period_3'] * 3.33333 - -data[
        'action_type_Running Bank shot'] > np.minimum(np.exp(-(1.27586 * 
        1.27586)), data['action_type_Jump Bank Shot'])).astype(float)))))) /
        2.0)) + np.tanh((data['action_type_Running Jump Shot'] + (data[
        'action_type_Alley Oop Layup shot'] / 2.0 - (data['last_moments_1'] -
        data['shot_zone_area_Center(C)']))) / 2.0 * np.maximum(np.tanh(data
        ['action_type_Alley Oop Layup shot'] * data[
        'shot_zone_basic_Backcourt']) * 2.0, np.tanh(3.33333 + (np.maximum(
        data['action_type_Alley Oop Layup shot'], 10.162331581115723) -
        data['action_type_Step Back Jump shot'])))) + np.tanh((data[
        'action_type_Dunk Shot'] > (np.exp(-((data['last_moments_1'] > np.
        exp(-(data['action_type_Reverse Layup Shot'] * data[
        'action_type_Reverse Layup Shot']))).astype(float) * (data[
        'last_moments_1'] > np.exp(-(data['action_type_Reverse Layup Shot'] *
        data['action_type_Reverse Layup Shot']))).astype(float))) <= data[
        'action_type_Turnaround Fadeaway shot']).astype(float)).astype(
        float) * 2.0 - data['last_moments_1'] + (np.maximum(data[
        'action_type_Turnaround Fadeaway shot'], data[
        'action_type_Jump Bank Shot'] * np.exp(-(0.31831 * 0.31831)) - np.
        exp(-(data['action_type_Floating Jump shot'] * data[
        'action_type_Floating Jump shot']))) + (data[
        'action_type_Turnaround Fadeaway shot'] + np.minimum(data[
        'last_moments_1'], data['season_2003-04'])) / 2.0)) + np.tanh(--
        data['action_type_Jump Shot'] * 2.0 * (data['distance_0'] - np.
        maximum(1.51724 + data['action_type_Jump Shot'], data[
        'opponent_TOR'] - 0.294118 * 2.0 - -((1.0 + data[
        'action_type_Driving Layup Shot']) / 2.0)) - 1.23188)) + np.tanh(
        2.9182798862457275 * np.maximum(data['action_type_Slam Dunk Shot'],
        (data['shot_zone_area_Right Side Center(RC)'] > ((0.170213 + -data[
        'distance_25']) / 2.0 + (data['shot_zone_basic_Left Corner 3'] > (
        data['season_1998-99'] > data['season_2014-15'] * 2.0).astype(float
        )).astype(float) / 2.0) / 2.0).astype(float)) - data['season_2014-15']
        ) + np.tanh(0.594937 - (data['minutes_remaining_0'] + ((np.exp(-(
        1.51724 * (0.170213 > (data['lat'] <= np.maximum(data['distance_0'],
        data['minutes_remaining_0'])).astype(float)).astype(float) * (
        1.51724 * (0.170213 > (data['lat'] <= np.maximum(data['distance_0'],
        data['minutes_remaining_0'])).astype(float)).astype(float)))) + np.
        tanh(data['action_type_Jump Bank Shot'] + np.tanh(0.470588))) / 2.0 <=
        -data['minutes_remaining_0']).astype(float))) + np.tanh(3.141593 *
        (data['opponent_POR'] + data['action_type_Jump Shot'] * ((data[
        'action_type_Layup Shot'] - 1.01449) * 2.0) * 2.0)) + np.tanh(np.
        maximum(data['shot_zone_area_Left Side Center(LC)'], np.maximum(
        data['action_type_Driving Layup Shot'], (data[
        'action_type_Slam Dunk Shot'] + (data[
        'action_type_Running Hook Shot'] - data['shot_zone_area_Center(C)']
        )) / 2.0)) - (data['shot_zone_basic_Restricted Area'] - (data[
        'shot_zone_range_Back Court Shot'] > data['distance_24'] * ((np.
        tanh((data['season_2006-07'] + data['action_type_Slam Dunk Shot'] +
        --0.0) / 2.0) + data['action_type_Driving Layup Shot']) / 2.0)).
        astype(float))) + np.tanh(-((data['season_2011-12'] + (data[
        'season_2015-16'] - ((data['opponent_PHX'] + np.maximum(data[
        'minutes_remaining_6'], np.minimum((data['opponent_PHX'] <= np.tanh
        (data['distance_8'])).astype(float), data['distance_41'] / 2.0 * 
        2.0) + data['action_type_Driving Finger Roll Layup Shot']) * np.exp
        (-(-(0.170213 * 2.0) / 2.0 * (-(0.170213 * 2.0) / 2.0)))) / 2.0 +
        data['opponent_PHX']) / 2.0 * 2.0)) / 2.0)) + np.tanh(data[
        'action_type_Jump Bank Shot'] + (np.maximum(np.maximum(data[
        'opponent_NYK'], data['action_type_Driving Finger Roll Shot']),
        data['action_type_Running Jump Shot']) + data['opponent_SEA']) / 
        2.0 / 2.0 + data['action_type_Running Jump Shot']) + np.tanh(np.
        minimum(0.170213, data['shot_zone_basic_Right Corner 3']) * 2.0 + 
        data['action_type_Alley Oop Dunk Shot'] / 2.0 + (data[
        'action_type_Reverse Dunk Shot'] + ((data[
        'action_type_Slam Dunk Shot'] + (data[
        'action_type_Fadeaway Bank shot'] + data[
        'action_type_Slam Dunk Shot']) / 2.0) / 2.0 - data['distance_7'])) /
        2.0) + np.tanh(((data['action_type_Jump Hook Shot'] * data[
        'shot_zone_area_Left Side(L)'] > data['lat']).astype(float) + (data
        ['action_type_Driving Layup Shot'] - np.minimum(data[
        'action_type_Turnaround Bank shot'] + 2.0, data['season_2015-16'] -
        -(np.minimum(data['distance_17'], data['action_type_Tip Shot'] * 
        2.0) - data['action_type_Running Hook Shot'])))) / 2.0 + data[
        'season_1999-00']) + np.tanh(-np.maximum(data['season_1997-98'], 
        data['distance_2'] * 2.0) * 2.0 - (data['action_type_Jump Shot'] - 
        np.maximum(data['action_type_Slam Dunk Shot'], data[
        'action_type_Driving Dunk Shot'] * 2.0) / 2.0)) + np.tanh(data[
        'last_moments_0'] + (data['distance_25'] + (((data['distance_15'] >
        -np.tanh(0.147727 / 2.0)).astype(float) + data[
        'action_type_Running Jump Shot']) / 2.0 + (data[
        'action_type_Slam Dunk Shot'] + data[
        'shot_zone_basic_Left Corner 3']))) / 2.0) + np.tanh(np.maximum(
        data['distance_19'], np.maximum(data['distance_20'], (data[
        'action_type_Driving Reverse Layup Shot'] + np.maximum(np.maximum(
        data['action_type_Running Bank shot'], data['season_2005-06'] * 2.0 *
        2.0), data['distance_23'])) / 2.0 / 2.0 + data[
        'action_type_Turnaround Jump Shot']))) + np.tanh(data[
        'action_type_Pullup Jump shot'] + ((data['shot_zone_range_24+ ft.'] +
        np.maximum(data['distance_21'], data['opponent_SAC'])) / 2.0 + (
        data['action_type_Finger Roll Layup Shot'] - data[
        'action_type_Jump Shot']) + data['shot_zone_range_24+ ft.']) / 2.0
        ) + np.tanh(((data['action_type_Pullup Jump shot'] + np.maximum((
        data['action_type_Dunk Shot'] + data['action_type_Pullup Jump shot'
        ]) / 2.0, data['distance_24'] + np.maximum(np.minimum(10.0, data[
        'action_type_Jump Bank Shot']), data['distance_23']) * 1.23188)) / 
        2.0 + (data['action_type_Dunk Shot'] + -data['distance_3']) / 2.0) /
        2.0) + np.tanh(data['action_type_Slam Dunk Shot'] + (((0.170213 / 
        2.0 <= data['shot_zone_range_16-24 ft.']).astype(float) + data[
        'shot_zone_range_16-24 ft.']) / 2.0 - (data[
        'shot_zone_range_Less Than 8 ft.'] - np.maximum(-data[
        'action_type_Jump Shot'], data['shot_zone_range_16-24 ft.'])))
        ) + np.tanh(np.maximum(data['shot_zone_range_8-16 ft.'], data[
        'action_type_Dunk Shot']) + (data['action_type_Dunk Shot'] + (data[
        'last_moments_0'] + (data['last_moments_0'] + np.minimum(np.maximum
        (2.0, 0.31831), data['away_False'] - data['action_type_Dunk Shot']) -
        np.tanh(data['distance_15']))) / 2.0) / 2.0) + np.tanh((data[
        'action_type_Reverse Dunk Shot'] + data[
        'action_type_Pullup Jump shot'] - np.maximum(data['distance_9'], -(
        data['last_moments_0'] + data['opponent_SEA'])) + -data[
        'action_type_Fadeaway Jump Shot']) / 2.0) + np.tanh((data[
        'distance_5'] - data['season_2007-08'] + -np.maximum(data[
        'action_type_Hook Shot'], 1.80645)) * (data['minutes_remaining_2'] >
        np.minimum((np.maximum(data['distance_9'], 1.27586) + (data[
        'action_type_Hook Shot'] + np.minimum(6.0, np.exp(-((data[
        'action_type_Driving Reverse Layup Shot'] + 0.0 / 2.0) * (data[
        'action_type_Driving Reverse Layup Shot'] + 0.0 / 2.0))))) / 2.0) /
        2.0, -(data['action_type_Hook Shot'] - data['distance_41']))).
        astype(float)) + np.tanh(3.33333 * ((data[
        'action_type_Reverse Dunk Shot'] + (data[
        'action_type_Slam Dunk Shot'] + (data['season_2007-08'] + np.
        maximum(data['action_type_Floating Jump shot'], (data[
        'action_type_Driving Slam Dunk Shot'] + (data['distance_5'] + data[
        'action_type_Alley Oop Dunk Shot'])) / 2.0)) / 2.0) / 2.0) / 2.0)
        ) + np.tanh(np.maximum(data['action_type_Driving Dunk Shot'], -data
        ['shot_zone_range_Less Than 8 ft.']) + -(data['season_2015-16'] - 
        data['distance_11'] * ((5.050303936004639 + data[
        'action_type_Driving Dunk Shot']) / 2.0 <= 0.594937).astype(float))
        ) + np.tanh((np.maximum(data['action_type_Running Jump Shot'], data
        ['distance_1']) - (data['shot_zone_area_Left Side(L)'] / 2.0 + (
        data['distance_13'] * 2.0 > (data['action_type_Floating Jump shot'] +
        data['distance_8'] * np.maximum(1.0, np.maximum(np.tanh(0.147727), 
        np.maximum(data['season_2015-16'], np.exp(-((0.147727 <= -data[
        'minutes_remaining_9']).astype(float) * (0.147727 <= -data[
        'minutes_remaining_9']).astype(float))) / 2.0) * 2.0))) / 2.0).
        astype(float)) / 2.0) * 2.0) + np.tanh((data[
        'action_type_Alley Oop Dunk Shot'] + (data[
        'action_type_Reverse Dunk Shot'] * 2.0 + (data['period_1'] + data[
        'action_type_Alley Oop Dunk Shot'] / 2.0 + data[
        'action_type_Fadeaway Bank shot']) / 2.0 * 2.0) + data[
        'minutes_remaining_7']) / 2.0) + np.tanh(np.minimum(np.maximum((
        data['action_type_Driving Dunk Shot'] + (data[
        'action_type_Driving Finger Roll Layup Shot'] + data[
        'action_type_Driving Slam Dunk Shot'] * 2.0) / 2.0) / 2.0 + data[
        'season_2000-01'] * (1.20833 * 2.0), data['action_type_Dunk']), np.
        minimum(np.tanh(np.maximum(data['action_type_Running Bank shot'], 
        1.27586)) * 2.0, 1.23188)) + data['action_type_Driving Dunk Shot']
        ) + np.tanh((data['action_type_Slam Dunk Shot'] + np.maximum(data[
        'distance_21'], (data['season_2008-09'] + ((np.exp(-(data[
        'opponent_VAN'] * data['opponent_VAN'])) + np.tanh(data[
        'last_moments_0'])) / 2.0 + (data['season_1998-99'] - -data[
        'distance_2']))) / 2.0 + data['action_type_Jump Bank Shot'] / 2.0 *
        2.0)) / 2.0) + np.tanh(np.minimum(data['distance_0'], 42.5) - (data
        ['action_type_Tip Shot'] + data['action_type_Layup Shot'] * 2.0)
        ) + np.tanh(data['shot_zone_range_Less Than 8 ft.'] - (data[
        'action_type_Layup Shot'] + np.maximum(data['distance_5'] * 2.0, (
        data['action_type_Layup Shot'] * 2.5 + (data['distance_5'] * 2.0 -
        np.tanh(data['action_type_Layup Shot']))) / 2.0))) + np.tanh((np.
        minimum(np.minimum(data['season_2012-13'], float(1.20833 > 1.79412 *
        -0.0) / 2.0), data['playoffs_1']) + ((data[
        'action_type_Slam Dunk Shot'] + data['distance_22']) / 2.0 - (data[
        'opponent_HOU'] - (data['action_type_Slam Dunk Shot'] + data[
        'action_type_Floating Jump shot'])))) / 2.0) + np.tanh(data[
        'shot_zone_basic_Restricted Area'] - (data['action_type_Layup Shot'
        ] * 2.0 - (np.minimum(0.170213, data['shot_zone_range_8-16 ft.'] * 
        2.0) > -(data['action_type_Layup Shot'] - data[
        'shot_zone_basic_Mid-Range'] * -((data['action_type_Layup Shot'] + 
        np.tanh(np.tanh((data['action_type_Driving Finger Roll Shot'] +
        data['action_type_Turnaround Bank shot']) / 2.0 / 2.0)) * 2.0) / 
        2.0 * 2.0))).astype(float))) + np.tanh((-((data['opponent_VAN'] + (
        data['distance_37'] + data['distance_6']) + (data[
        'action_type_Layup'] - data['distance_5'])) / 2.0 / 2.0) + data[
        'away_False'] * np.maximum(np.maximum(data['opponent_NJN'], data[
        'action_type_Running Jump Shot']), np.minimum(data[
        'shot_zone_basic_Above the Break 3'], 1.80645))) / 2.0) + np.tanh(
        data['action_type_Jump Shot'] * np.tanh(data['away_True'] - -data[
        'action_type_Tip Shot'])) + np.tanh(0.490566 * (data[
        'shot_zone_area_Center(C)'] * np.maximum(data[
        'shot_zone_range_8-16 ft.'], np.tanh(0.170213)) + (data[
        'action_type_Alley Oop Dunk Shot'] + data['distance_24']))) + np.tanh(
        -(data['action_type_Driving Layup Shot'] + np.maximum((data[
        'action_type_Jump Shot'] + (np.minimum(1.27586 + data['distance_26'
        ], 2.5 * 0.0) <= np.tanh(data['action_type_Layup Shot'] - (0.490566 >
        data['action_type_Alley Oop Layup shot']).astype(float))).astype(
        float)) / 2.0, data['action_type_Layup Shot'] + -data[
        'last_moments_0']))) + np.tanh(data['lat'] * ((data[
        'action_type_Running Jump Shot'] > (0.147727 <= ((1.0 - 0.594937) *
        2.0 > np.minimum((np.maximum(data['action_type_Driving Layup Shot'],
        data['action_type_Driving Dunk Shot']) + data[
        'action_type_Driving Dunk Shot']) / 2.0, np.exp(-(1.51724 * 1.51724
        )))).astype(float)).astype(float)).astype(float) + np.maximum(data[
        'action_type_Driving Dunk Shot'], data['minutes_remaining_2']))
        ) + np.tanh(np.minimum(-((data['action_type_Layup Shot'] + data[
        'action_type_Tip Shot']) / 2.0), 0.147727) - (data['last_moments_1'
        ] + (data['action_type_Layup Shot'] + np.maximum(data[
        'last_moments_1'], data['opponent_OKC'] - -data[
        'action_type_Tip Shot'])) / 2.0)) + np.tanh((data[
        'action_type_Pullup Jump shot'] + (np.maximum(data[
        'action_type_Fadeaway Bank shot'], np.maximum(data[
        'action_type_Running Hook Shot'], np.minimum(data[
        'action_type_Running Bank shot'], (3.0 + data[
        'action_type_Slam Dunk Shot']) / 2.0) * 2.0) * 2.0) + (data[
        'action_type_Slam Dunk Shot'] + (data['opponent_NYK'] + data[
        'action_type_Dunk']) / 2.0)) / 2.0) / 2.0) + np.tanh((data[
        'distance_0'] + (data['season_2005-06'] + np.maximum(data[
        'distance_0'] * 2.0 / 2.0, np.maximum(data['distance_1'], data[
        'distance_23'])))) / 2.0) + np.tanh(np.maximum(data['distance_2'], 
        data['shot_zone_basic_In The Paint (Non-RA)'] * ((data[
        'last_moments_1'] + -(data['action_type_Driving Reverse Layup Shot'
        ] + (data['action_type_Slam Dunk Shot'] > np.exp(-((data[
        'distance_29'] + (1.80645 > data['distance_29']).astype(float)) / 
        2.0 * ((data['distance_29'] + (1.80645 > data['distance_29']).
        astype(float)) / 2.0)))).astype(float) > data['distance_27'] - np.
        exp(-(data['action_type_Floating Jump shot'] * 2.0 * (data[
        'action_type_Floating Jump shot'] * 2.0)))).astype(float)) / 2.0 *
        (data['action_type_Jump Shot'] * 1.20833)))) + np.tanh(--(data[
        'distance_1'] - ((data['away_False'] - data['action_type_Dunk'] / 
        2.0) * (data['distance_0'] * ((data['action_type_Jump Hook Shot'] +
        np.exp(-(data['period_6'] * data['period_6']))) * 2.0)) + data[
        'distance_1']) / 2.0)) + np.tanh(data['opponent_HOU'] * (data[
        'away_False'] - (data['action_type_Running Hook Shot'] + data[
        'action_type_Running Bank shot'] * ((float(1.20833 <= 1.20833) +
        data['action_type_Finger Roll Layup Shot']) / 2.0) + data[
        'action_type_Jump Bank Shot']))) + np.tanh((data[
        'action_type_Slam Dunk Shot'] - (data['action_type_Tip Shot'] +
        data['action_type_Slam Dunk Shot']) / 2.0 + (data[
        'action_type_Slam Dunk Shot'] > np.minimum(3.33333, 1.20833 + data[
        'lon'])).astype(float)) / 2.0) + np.tanh((data['season_2006-07'] + 
        (data['action_type_Alley Oop Dunk Shot'] * 1.80645 + np.maximum(
        data['action_type_Dunk'], data['action_type_Running Bank shot'] +
        data['action_type_Pullup Jump shot'])) / 2.0) / 2.0) + np.tanh(-((
        data['shot_zone_area_Center(C)'] * data['season_2002-03'] - (data[
        'action_type_Alley Oop Dunk Shot'] + np.minimum(data['playoffs_1'] -
        data['shot_zone_area_Back Court(BC)'], (float(1.80645 / 2.0 > 1.0) +
        (np.exp(-(np.minimum(data['action_type_Jump Hook Shot'], 
        9.77937126159668) * np.minimum(data['action_type_Jump Hook Shot'], 
        9.77937126159668))) > 0.170213 - data['opponent_CHI']).astype(float
        )) / 2.0)) / 2.0) / 2.0)) + np.tanh((data['opponent_VAN'] + (-data[
        'opponent_ORL'] + (data['shot_zone_basic_Left Corner 3'] + data[
        'minutes_remaining_5'] / 2.0 * np.maximum(data['season_2003-04'] + 
        6.0625, np.exp(-(np.exp(-(np.exp(-(0.147727 * 0.147727)) * np.exp(-
        (0.147727 * 0.147727)))) * np.exp(-(np.exp(-(0.147727 * 0.147727)) *
        np.exp(-(0.147727 * 0.147727))))))) * (np.tanh((0.147727 + 0.472727
        ) / 2.0) - data['action_type_Turnaround Bank shot']))) / 2.0) / 2.0
        ) + np.tanh(data['distance_7'] * data['season_2007-08'] + (1.51724 <=
        np.maximum(data['lon'], np.minimum(0.9912230968475342, (data[
        'distance_12'] > 1.20833).astype(float)))).astype(float)) + np.tanh(
        (data['action_type_Driving Dunk Shot'] + data[
        'shot_zone_range_16-24 ft.'] * np.tanh(np.minimum(-data[
        'action_type_Running Hook Shot'], data['away_False']) - data[
        'season_2007-08'])) / 2.0) + np.tanh(data[
        'action_type_Fadeaway Bank shot'] + (((data[
        'action_type_Driving Finger Roll Shot'] + np.maximum(data[
        'shot_zone_area_Left Side Center(LC)'], np.minimum(data[
        'action_type_Driving Dunk Shot'], np.exp(-(data[
        'action_type_Driving Dunk Shot'] * data[
        'action_type_Driving Dunk Shot']))) * (data['minutes_remaining_6'] <=
        data['distance_31']).astype(float)) / 2.0 / 2.0) / 2.0 + (data[
        'minutes_remaining_6'] + data['action_type_Driving Dunk Shot']) / 
        2.0) / 2.0 - (data['action_type_Tip Shot'] + data['opponent_POR'] /
        2.0) / 2.0)) + np.tanh(data['action_type_Fadeaway Jump Shot'] * np.
        exp(-(data['distance_4'] * 2.0 * (data['distance_4'] * 2.0))) * (
        data['action_type_Reverse Layup Shot'] + (data['season_2001-02'] -
        np.maximum(data['action_type_Alley Oop Dunk Shot'], data[
        'action_type_Driving Finger Roll Layup Shot'])) / 2.0)) + np.tanh((
        (data['action_type_Driving Finger Roll Layup Shot'] + (data[
        'shot_zone_range_16-24 ft.'] + data['distance_15'])) * 2.0 + data[
        'minutes_remaining_2'] > data['action_type_Driving Dunk Shot']).
        astype(float) + data['action_type_Slam Dunk Shot'] + data[
        'opponent_DET'] * (1.20833 / 2.0) / 2.0 + (data[
        'action_type_Driving Dunk Shot'] + data[
        'action_type_Running Hook Shot']) / 2.0) + np.tanh(data[
        'minutes_remaining_6'] * ((data['season_2014-15'] + (data[
        'action_type_Tip Shot'] - (data['action_type_Running Bank shot'] +
        (data['action_type_Driving Slam Dunk Shot'] + -((np.maximum(data[
        'opponent_DAL'], data['distance_8']) <= -(np.tanh(data['distance_8'
        ]) / 2.0)).astype(float) * data['distance_45']))) / 2.0)) / 2.0)
        ) + np.tanh((data['action_type_Slam Dunk Shot'] - (-data[
        'opponent_NOH'] + data['opponent_POR']) / 2.0 + np.maximum(data[
        'distance_2'], np.maximum(data['action_type_Driving Dunk Shot'], np
        .maximum(data['action_type_Driving Dunk Shot'] - data['distance_34'
        ], np.minimum(data['season_2003-04'], 0.490566)) * 2.0))) * (data[
        'shot_zone_basic_Restricted Area'] > 0.294118).astype(float) / 2.0
        ) + np.tanh(-(data['shot_zone_basic_Restricted Area'] * np.tanh(
        data['minutes_remaining_4'] + (data['season_2001-02'] > data[
        'distance_26'] * (1.01449 / 2.0 - np.minimum((data['season_2007-08'
        ] + np.minimum(data['action_type_Hook Shot'], data['season_2001-02'
        ])) / 2.0 * data['action_type_Reverse Layup Shot'], (data[
        'distance_45'] <= 1.27586).astype(float) * 0.31831))).astype(float)))
        ) + np.tanh(np.maximum((data['action_type_Alley Oop Dunk Shot'] -
        data['shot_zone_range_Less Than 8 ft.'] - data[
        'action_type_Alley Oop Dunk Shot'] * data['distance_30']) / 2.0, 
        data['action_type_Running Bank shot'] * np.exp(-(np.minimum(
        0.886364 * 2.0 * 2.0 * 2.0 / 2.0 * (data['opponent_ORL'] <= data[
        'opponent_ORL']).astype(float) / 2.0, data[
        'shot_zone_range_Less Than 8 ft.']) * np.minimum(0.886364 * 2.0 * 
        2.0 * 2.0 / 2.0 * (data['opponent_ORL'] <= data['opponent_ORL']).
        astype(float) / 2.0, data['shot_zone_range_Less Than 8 ft.'])))) +
        (data['action_type_Slam Dunk Shot'] + data[
        'action_type_Driving Dunk Shot'])) + np.tanh(((data[
        'action_type_Turnaround Bank shot'] * 2.0 - (data['distance_38'] >
        data['distance_34']).astype(float)) * 2.0 * np.exp(-(np.minimum(
        1.79412 * 2.0 * 2.0, data['minutes_remaining_8'] * 2.0) * np.
        minimum(1.79412 * 2.0 * 2.0, data['minutes_remaining_8'] * 2.0))) -
        data['minutes_remaining_8'] + data['distance_24']) / 2.0 / 2.0
        ) + np.tanh(data['action_type_Jump Shot'] * ((data[
        'action_type_Turnaround Bank shot'] + np.maximum(data[
        'action_type_Jump Hook Shot'] * data['opponent_TOR'], 0.886364) * 
        2.0) * 2.0 / 2.0) * data['opponent_TOR']) + np.tanh((data[
        'action_type_Alley Oop Dunk Shot'] - data['season_2003-04'] * ((-(
        data['action_type_Hook Shot'] > (data['opponent_VAN'] + np.exp(-(
        data['action_type_Alley Oop Dunk Shot'] * data[
        'action_type_Alley Oop Dunk Shot']))) / 2.0).astype(float) > (data[
        'distance_16'] + data['action_type_Alley Oop Dunk Shot']) / 2.0).
        astype(float) / 2.0)) * np.exp(-(np.tanh(0.147727) * np.tanh(
        0.147727))) - np.maximum(data['action_type_Layup Shot'], np.maximum
        (data['shot_zone_basic_Backcourt'], data['action_type_Hook Shot']))
        ) + np.tanh(data['away_False'] * ((np.exp(-(np.exp(-(data[
        'action_type_Driving Finger Roll Shot'] * data[
        'action_type_Driving Finger Roll Shot'])) * np.exp(-(data[
        'action_type_Driving Finger Roll Shot'] * data[
        'action_type_Driving Finger Roll Shot'])))) + -np.maximum(np.
        maximum(data['shot_zone_basic_Restricted Area'], np.minimum(data[
        'action_type_Finger Roll Layup Shot'], 2.5)), np.minimum(data[
        'action_type_Driving Jump shot'], (data[
        'action_type_Driving Finger Roll Shot'] - 1.51724) / 2.0)) + -data[
        'action_type_Jump Shot']) / 2.0)) + np.tanh(-((data[
        'minutes_remaining_11'] + (data['distance_29'] > (2.5 <= data[
        'period_1'] - np.tanh(data['action_type_Dunk Shot'])).astype(float)
        ).astype(float)) / 2.0) - -data['action_type_Dunk Shot'] * (data[
        'away_True'] - np.maximum(data['shot_zone_basic_Left Corner 3'], 
        0.147727 - 0.886364))) + np.tanh(data['opponent_DEN'] * ((data[
        'action_type_Jump Shot'] + data['action_type_Jump Shot'] * (data[
        'season_2009-10'] + np.maximum(data['season_1999-00'], np.maximum(
        data['shot_zone_area_Right Side Center(RC)'], (3.33333 > np.maximum
        (data['shot_zone_basic_Right Corner 3'] + data['opponent_GSW'] * 
        2.0, data['opponent_NOP'])).astype(float) + data['distance_3'])))) /
        2.0)) + np.tanh(data['action_type_Driving Slam Dunk Shot'] + (data[
        'action_type_Slam Dunk Shot'] - np.maximum(-np.minimum(3.33333,
        data['action_type_Slam Dunk Shot']) / 2.0, data[
        'action_type_Driving Layup Shot'])) / 2.0) + np.tanh(data[
        'minutes_remaining_3'] * -((np.maximum(np.tanh(data[
        'shot_zone_basic_Mid-Range']), data['season_2001-02'] * 2.0) + np.
        maximum(data['opponent_SAC'], np.tanh(np.tanh(data[
        'shot_zone_basic_Mid-Range'] * ((data[
        'action_type_Running Jump Shot'] + -(data['playoffs_1'] > (data[
        'action_type_Floating Jump shot'] + 0.490566) / 2.0).astype(float)) /
        2.0)))) / 2.0 * 2.0) / 2.0)) + np.tanh(data['away_True'] * ((np.
        maximum(np.minimum(data['action_type_Driving Dunk Shot'], (np.exp(-
        (data['opponent_CHI'] * data['opponent_CHI'])) + data[
        'opponent_MIN']) / 2.0), data['season_2009-10']) + data[
        'action_type_Layup Shot'] / 2.0) / 2.0)) + np.tanh(((np.exp(-(0.0 *
        0.0)) + data['opponent_DAL']) / 2.0 <= data['minutes_remaining_0'] +
        (np.minimum(0.31831, data['minutes_remaining_10'] * np.tanh(data[
        'distance_25']) / 2.0 + data['opponent_DAL']) <= np.minimum(data[
        'minutes_remaining_9'], np.tanh(np.exp(-(data['distance_25'] * data
        ['distance_25']))))).astype(float)).astype(float)) + np.tanh(data[
        'action_type_Floating Jump shot'] + np.exp(-(data['season_2011-12'] *
        data['season_2011-12'])) * (data['action_type_Jump Bank Shot'] +
        data['action_type_Fadeaway Bank shot']) / 2.0) + np.tanh((data[
        'action_type_Dunk Shot'] - data['shot_zone_range_24+ ft.']) * np.
        maximum(data['action_type_Slam Dunk Shot'], (data['away_True'] + np
        .exp(-(data['distance_39'] * data['distance_39'])) / 2.0) / 2.0)
        ) + np.tanh(np.minimum(data['action_type_Alley Oop Dunk Shot'], np.
        minimum((-1.27586 * (data['opponent_BOS'] / 2.0) + np.exp(-(np.exp(
        -((-(1.01449 > data['action_type_Dunk']).astype(float) > 1.0).
        astype(float) * (-(1.01449 > data['action_type_Dunk']).astype(float
        ) > 1.0).astype(float))) * np.exp(-((-(1.01449 > data[
        'action_type_Dunk']).astype(float) > 1.0).astype(float) * (-(
        1.01449 > data['action_type_Dunk']).astype(float) > 1.0).astype(
        float)))))) / 2.0, -data['action_type_Tip Shot']) - data[
        'action_type_Turnaround Jump Shot'])) + np.tanh(data['opponent_GSW'
        ] * (data['season_2001-02'] + np.minimum((np.exp(-(0.594937 * data[
        'distance_32'] * (0.594937 * data['distance_32']))) + (np.tanh((max
        (1.0, (3.33333 > np.exp(-(1.80645 * 1.80645))).astype(float)) + (
        0.470588 + 0.0 / 2.0) / 2.0 > data['season_2014-15']).astype(float)
        ) + data['distance_14']) / 2.0) / 2.0, data['season_1996-97']))
        ) + np.tanh((data['action_type_Hook Shot'] + np.maximum(data[
        'action_type_Tip Shot'], np.minimum(data['distance_27'] * np.tanh((
        data['playoffs_0'] + float(42.5 <= 1.20833) * 2.0) / 2.0), 1.80645 *
        np.exp(-(data['action_type_Jump Hook Shot'] * data[
        'action_type_Jump Hook Shot']))))) * (-2.5 / 2.0)) + np.tanh(-(np.
        maximum(data['action_type_Running Jump Shot'] + data[
        'action_type_Hook Shot'] * np.minimum(data[
        'action_type_Driving Finger Roll Shot'], np.tanh(np.tanh(1.23188)) -
        (1.51724 <= (float(1.0 <= 0.147727) / 2.0 + data['opponent_WAS']) /
        2.0).astype(float)), data['opponent_NJN'] + (data[
        'action_type_Hook Shot'] - (data['distance_32'] <= 0.170213).astype
        (float))) / 2.0)) + np.tanh(((data['opponent_SAC'] + np.tanh(data[
        'distance_24'])) * 2.0 + np.minimum(data['opponent_DAL'], np.
        minimum((1.80645 + data['distance_38']) / 2.0, -(0.472727 * -0.0))) *
        (((np.exp(-((0.470588 + data['distance_41']) * (0.470588 + data[
        'distance_41']))) + data['distance_38']) / 2.0 + 1.0 / 2.0) / 2.0)) /
        2.0 * data['season_2015-16']) + np.tanh((data[
        'action_type_Driving Dunk Shot'] + (data[
        'action_type_Running Jump Shot'] + np.minimum(data['period_1'] * np
        .minimum(data['opponent_SEA'], -np.exp(-(data[
        'minutes_remaining_11'] * data['minutes_remaining_11'])) / 2.0),
        data['distance_40'])) * ((data['away_False'] + (0.846154 + data[
        'opponent_CHI']) / 2.0 / 2.0) / 2.0)) / 2.0) + np.tanh((0.0 - np.
        minimum(data['action_type_Tip Shot'], 6.0625)) / 2.0 + np.minimum((
        data['action_type_Reverse Dunk Shot'] <= 1.01449).astype(float), 
        data['away_False'] * --np.maximum(data['season_2006-07'], data[
        'action_type_Alley Oop Dunk Shot']))) + np.tanh(data[
        'season_1999-00'] * (data['shot_zone_basic_Mid-Range'] + data[
        'action_type_Layup Shot'])) + np.tanh((np.maximum(data[
        'action_type_Pullup Jump shot'], data['lat']) > np.minimum(3.0, 
        1.79412 / 2.0)).astype(float)) + np.tanh((np.minimum(data[
        'distance_3'], np.minimum(data['distance_32'] * (data['period_4'] *
        data['distance_27']) / 2.0, data['distance_19'])) + (data[
        'action_type_Slam Dunk Shot'] + np.tanh(data['distance_19']) + (
        data['action_type_Reverse Dunk Shot'] + data['season_2008-09'])) / 
        2.0) / 2.0) + np.tanh(data['season_2012-13'] * (data[
        'shot_zone_range_Back Court Shot'] + (data['distance_0'] - data[
        'distance_14']) * (0.31831 * 1.79412 > np.tanh(0.0 * data[
        'distance_2'])).astype(float))) + np.tanh(data['last_moments_1'] *
        np.minimum(data['shot_type_2PT Field Goal'], -(data[
        'season_2010-11'] + data['action_type_Slam Dunk Shot']))) + np.tanh(
        (data['action_type_Driving Dunk Shot'] - data[
        'action_type_Tip Shot'] + (data['period_2'] + (data['distance_22'] *
        np.tanh(data['action_type_Driving Dunk Shot']) > 1.27586 - 1.80645)
        .astype(float) > 1.80645).astype(float)) / 2.0) + np.tanh(data[
        'action_type_Running Hook Shot'] - data['season_2007-08'] * np.
        minimum(np.minimum(data['action_type_Layup Shot'], 1.01449), 2.5)
        ) + np.tanh(data['action_type_Turnaround Bank shot'] + np.minimum(
        data['season_2014-15'] * (np.minimum((data['season_2005-06'] > 42.5
        ).astype(float), (data['distance_10'] + 1.79412) / 2.0) + data[
        'opponent_SAC']) * 2.0 * data['minutes_remaining_0'], 2.5)) + np.tanh(
        (data['shot_zone_basic_Restricted Area'] + -np.maximum(data[
        'season_2000-01'], (data['season_2001-02'] + np.tanh(data[
        'distance_27'])) / 2.0) * 2.0) / 2.0 * data['season_2011-12']
        ) + np.tanh(data['shot_zone_range_Less Than 8 ft.'] * np.maximum(
        data['minutes_remaining_10'], np.tanh((3.33333 * 2.0 <= data[
        'opponent_ATL'] - data['action_type_Alley Oop Dunk Shot']).astype(
        float)))) + np.tanh(data['shot_zone_area_Center(C)'] * (data[
        'season_2011-12'] > np.tanh(np.minimum(np.tanh(data['opponent_CLE']
        ), np.tanh(np.exp(-(data['distance_44'] * data['distance_44'])))) *
        np.maximum(data['season_2010-11'], np.tanh(np.minimum(42.5, data[
        'distance_20']))))).astype(float)) + np.tanh(np.minimum((data[
        'shot_zone_basic_Above the Break 3'] <= np.maximum(0.846154, data[
        'opponent_CHA']) * 2.0).astype(float), data['away_True'] + data[
        'minutes_remaining_7']) * np.minimum(data['action_type_Jump Shot'],
        np.tanh(data['season_1999-00'] + float(1.79412 > -1.79412)))
        ) + np.tanh(-(np.maximum(data['shot_zone_range_16-24 ft.'], data[
        'opponent_DAL']) * ((data['away_True'] + -np.exp(-(np.exp(-(((data[
        'season_2009-10'] > 0.470588 + data[
        'action_type_Running Layup Shot'] * 2.0).astype(float) > (0.0 / 2.0 +
        (1.01449 + np.exp(-(0.470588 * 0.470588))) / 2.0) / 2.0).astype(
        float) * ((data['season_2009-10'] > 0.470588 + data[
        'action_type_Running Layup Shot'] * 2.0).astype(float) > (0.0 / 2.0 +
        (1.01449 + np.exp(-(0.470588 * 0.470588))) / 2.0) / 2.0).astype(
        float))) * np.exp(-(((data['season_2009-10'] > 0.470588 + data[
        'action_type_Running Layup Shot'] * 2.0).astype(float) > (0.0 / 2.0 +
        (1.01449 + np.exp(-(0.470588 * 0.470588))) / 2.0) / 2.0).astype(
        float) * ((data['season_2009-10'] > 0.470588 + data[
        'action_type_Running Layup Shot'] * 2.0).astype(float) > (0.0 / 2.0 +
        (1.01449 + np.exp(-(0.470588 * 0.470588))) / 2.0) / 2.0).astype(
        float)))))) / 2.0))) + np.tanh(data['away_True'] * ((data[
        'distance_0'] + (-((0.0 + data['opponent_UTA']) / 2.0) - (np.
        minimum(0.470588, 0.0) > 1.23188).astype(float))) / 2.0)) + np.tanh(
        (data['action_type_Slam Dunk Shot'] + data['distance_26'] * (np.
        minimum(data['distance_10'], 42.5 - data['away_False']) - (0.31831 *
        2.0 <= (np.tanh(data['distance_15']) <= np.exp(-(data[
        'action_type_Hook Shot'] * data['action_type_Hook Shot'])) - data[
        'distance_34']).astype(float)).astype(float) * (data['away_True'] *
        2.0))) / 2.0) + np.tanh(data['season_2011-12'] * ((data[
        'action_type_Layup Shot'] + np.maximum(0.170213 * np.maximum(np.
        tanh(np.tanh(data['distance_36'] * (data['season_2011-12'] * (
        0.490566 > np.tanh(--(data['shot_zone_area_Center(C)'] + data[
        'playoffs_1']))).astype(float)))), data[
        'action_type_Driving Reverse Layup Shot']), data[
        'action_type_Jump Shot'])) / 2.0)) + np.tanh(np.maximum(data[
        'action_type_Dunk'], np.maximum(np.tanh((1.27586 <= (data[
        'distance_25'] + np.tanh(-np.minimum(data['opponent_SAC'], data[
        'distance_7'] + data['season_1997-98']))) / 2.0).astype(float)), (
        data['action_type_Slam Dunk Shot'] + data[
        'action_type_Driving Finger Roll Shot'] / 2.0) / 2.0 / 2.0))
        ) + np.tanh(np.maximum(np.maximum(data['opponent_MIL'], (data[
        'action_type_Driving Dunk Shot'] + np.maximum(data[
        'action_type_Driving Dunk Shot'], np.minimum(0.147727, np.exp(-((-
        np.maximum(data['period_2'], np.exp(-(0.170213 * 0.170213))) - data
        ['distance_4']) * (-np.maximum(data['period_2'], np.exp(-(0.170213 *
        0.170213))) - data['distance_4'])))))) * 2.0), data['opponent_NYK'] *
        (0.490566 > data['period_2'] - 0.470588).astype(float))) + np.tanh(np
        .minimum(3.0 * 2.0 * 2.5, data['season_1997-98'] * ((-data[
        'action_type_Alley Oop Dunk Shot'] + (data['action_type_Jump Shot'] -
        data['opponent_SAC'])) / 2.0))) + np.tanh((data[
        'action_type_Driving Dunk Shot'] + (data[
        'action_type_Slam Dunk Shot'] + data['distance_0'] * ((data[
        'action_type_Driving Finger Roll Shot'] + (data[
        'action_type_Reverse Dunk Shot'] + data[
        'action_type_Reverse Dunk Shot'])) / 2.0))) / 2.0) + np.tanh((data[
        'action_type_Dunk'] + data['away_True'] * ((data['distance_14'] + (
        data['action_type_Dunk Shot'] + data[
        'action_type_Floating Jump shot']) / 2.0) / 2.0)) / 2.0) + np.tanh(
        -((data['season_2002-03'] + data['distance_27']) * np.minimum(data[
        'action_type_Layup Shot'], ((data['distance_1'] > -np.minimum(-((
        data['distance_1'] + data['action_type_Slam Dunk Shot']) / 2.0), 
        0.594937)).astype(float) > np.minimum(0.886364, np.maximum(data[
        'distance_1'], data['opponent_LAC']))).astype(float)) / 2.0)
        ) + np.tanh(data['action_type_Jump Shot'] * -(data[
        'action_type_Fadeaway Bank shot'] - 2.0 * ((data['opponent_HOU'] +
        data['action_type_Hook Shot']) / 2.0))) + np.tanh(data[
        'shot_zone_range_Less Than 8 ft.'] * ((data['distance_28'] + np.
        minimum(np.exp(-(data['lat'] * data['lat'])) * -2.0, data[
        'action_type_Fadeaway Bank shot'])) / 2.0 / 2.0)) + np.tanh(data[
        'season_2012-13'] * (((data['shot_zone_area_Right Side(R)'] <= (np.
        exp(-(data['shot_zone_area_Right Side(R)'] * data[
        'shot_zone_area_Right Side(R)'])) <= data[
        'action_type_Driving Finger Roll Shot']).astype(float)).astype(
        float) + np.minimum((data['shot_zone_basic_Right Corner 3'] - np.
        minimum(-10.0 * 2.0, 42.5)) * data['action_type_Running Layup Shot'
        ] / 2.0, 0.886364 * np.tanh(data['distance_5'] - data['distance_15'
        ]))) / 2.0)) + np.tanh((data['season_2014-15'] * (data[
        'minutes_remaining_4'] * 3.33333) + np.maximum(np.minimum(data[
        'action_type_Driving Slam Dunk Shot'] / 2.0, data['period_7']), np.
        maximum(data['action_type_Driving Finger Roll Shot'], data[
        'action_type_Running Hook Shot']))) / 2.0) + np.tanh(np.tanh(data[
        'distance_20']) * data['opponent_TOR'] - np.exp(-(data[
        'season_2009-10'] * np.tanh(0.147727) * (data['season_2009-10'] *
        np.tanh(0.147727)))) * (data['shot_zone_range_Back Court Shot'] * 
        2.0 * 2.0) / 2.0 / 2.0 / 2.0) + np.tanh(data['season_2008-09'] * (
        data['away_True'] * np.minimum(data[
        'action_type_Finger Roll Layup Shot'], 1.01449 * np.tanh((data[
        'minutes_remaining_6'] - (data['distance_27'] / 2.0 + np.tanh(-(
        data['action_type_Hook Shot'] * np.tanh(1.27586 * 2.0)))) / 2.0) * 
        2.0) / 2.0) * 2.0) / 2.0) + np.tanh(np.maximum(data[
        'action_type_Driving Dunk Shot'] * 2.0, (data[
        'action_type_Driving Dunk Shot'] - (-data['minutes_remaining_0'] - 
        data['period_6'] * (data['opponent_MIL'] <= (data[
        'action_type_Driving Dunk Shot'] <= np.minimum(1.27586, data[
        'action_type_Driving Dunk Shot'])).astype(float)).astype(float))) *
        data['distance_16'])) + np.tanh((data['distance_15'] + data[
        'distance_18']) / 2.0 * -np.tanh(np.maximum(data[
        'action_type_Turnaround Jump Shot'], np.minimum(3.141593, data[
        'distance_20'])) / 2.0) * 2.0) + np.tanh(np.minimum(data['lat'] * (
        data['season_2010-11'] - data['distance_3']), (data[
        'action_type_Running Bank shot'] <= 1.27586 * data[
        'action_type_Jump Hook Shot'] - np.minimum(3.33333, 0.886364)).
        astype(float))) + np.tanh(data['minutes_remaining_9'] * (data[
        'action_type_Driving Finger Roll Shot'] / 2.0 + (-np.tanh(np.tanh(
        data['opponent_POR'])) + (data['action_type_Driving Jump shot'] - (
        data['shot_zone_area_Center(C)'] + data['opponent_POR']))))) + np.tanh(
        np.maximum(data['action_type_Finger Roll Layup Shot'], (data[
        'action_type_Driving Dunk Shot'] > np.minimum(np.tanh(0.846154), 
        data['action_type_Bank Shot'] * (data['opponent_DAL'] * 2.0))).
        astype(float) * 3.141593)) + np.tanh((data[
        'action_type_Alley Oop Dunk Shot'] + np.maximum(data[
        'action_type_Driving Reverse Layup Shot'], data['distance_1'] + np.
        maximum(data['action_type_Driving Dunk Shot'], ((data[
        'action_type_Fadeaway Bank shot'] + np.maximum(data[
        'action_type_Driving Dunk Shot'], (data[
        'action_type_Fadeaway Bank shot'] <= np.minimum(0.490566, (0.490566 <=
        data['distance_1']).astype(float) * data[
        'action_type_Alley Oop Dunk Shot'])).astype(float) / 2.0)) / 2.0 +
        (1.01449 <= data['action_type_Dunk']).astype(float)) / 2.0))) / 2.0
        ) + np.tanh(data['season_2008-09'] * (-(data['period_2'] * (np.
        minimum(data['season_2011-12'], np.tanh((data['distance_5'] > 
        0.31831).astype(float)) / 2.0) - data['opponent_POR'])) / 2.0)
        ) + np.tanh(np.maximum(data['action_type_Reverse Dunk Shot'], data[
        'opponent_PHI']) * np.tanh(data['distance_43'] * np.maximum(data[
        'distance_40'], np.maximum(3.141593, np.exp(-(np.tanh(data[
        'minutes_remaining_2']) * np.tanh(data[
        'action_type_Alley Oop Dunk Shot']) * (np.tanh(data[
        'minutes_remaining_2']) * np.tanh(data[
        'action_type_Alley Oop Dunk Shot'])))))) - data[
        'action_type_Jump Shot'] * ((data['distance_4'] > np.tanh(data[
        'minutes_remaining_9'] * 2.0)).astype(float) + np.tanh(np.tanh(
        0.846154))))) + np.tanh(data['minutes_remaining_9'] * data[
        'action_type_Jump Shot'] * ((data['shot_zone_range_24+ ft.'] - np.
        maximum((data['distance_24'] > data[
        'action_type_Driving Finger Roll Shot']).astype(float), np.maximum(
        data['distance_25'], -data['minutes_remaining_1']) * 2.0)) / 2.0)
        ) + np.tanh(1.51724 * (data['period_4'] * (np.maximum(data[
        'minutes_remaining_7'], np.maximum(data['distance_40'], (data[
        'distance_16'] > np.tanh(data['distance_39']) * 2.0).astype(float))
        ) / 2.0 / 2.0) * 2.0)) + np.tanh(data['shot_zone_range_8-16 ft.'] *
        (data['minutes_remaining_2'] / 2.0 * (np.maximum(data['period_2'],
        np.minimum((data['action_type_Slam Dunk Shot'] + (data[
        'distance_24'] > -((data['opponent_MIA'] + data['opponent_MIN']) / 
        2.0)).astype(float)) / 2.0 / 2.0, -(3.0 > (data['season_2000-01'] >
        1.23188).astype(float) / 2.0).astype(float))) + (data['distance_24'
        ] - data['action_type_Pullup Jump shot'])))) + np.tanh(data[
        'action_type_Driving Slam Dunk Shot'] + 0.470588 * ((np.maximum(
        data['distance_43'], data['season_2006-07']) + -(data['distance_29'
        ] * 2.0 + (data['distance_31'] + (0.170213 + np.minimum(data[
        'opponent_NJN'], np.minimum(data['action_type_Running Hook Shot'], 
        (0.490566 + 0.846154) / 2.0)) / 2.0)) / 2.0)) / 2.0)) + np.tanh(np.
        maximum(data['action_type_Driving Dunk Shot'], (data[
        'shot_zone_basic_Restricted Area'] + np.exp(-(3.0 * 3.0))) / 2.0 *
        np.maximum(data['action_type_Turnaround Fadeaway shot'], data[
        'period_3'] - np.maximum(data['shot_zone_range_24+ ft.'], -(np.
        minimum(np.minimum(3.33333, data['season_2006-07']), np.tanh(np.
        tanh(data['action_type_Alley Oop Dunk Shot']))) * 2.0) / 2.0 / 2.0)))
        ) + np.tanh(data['opponent_GSW'] * (data['away_True'] * data[
        'action_type_Driving Layup Shot'])) + np.tanh(data['opponent_MIL'] *
        ((data['shot_zone_range_Less Than 8 ft.'] + (data['distance_25'] +
        (data['action_type_Driving Slam Dunk Shot'] > np.tanh(0.294118)).
        astype(float)) / 2.0) / 2.0)) + np.tanh(np.tanh(data[
        'shot_zone_range_8-16 ft.'] * np.maximum(data['period_5'], (-np.
        minimum(data['opponent_DET'] * (6.0625 * 2.0 / 2.0 * data[
        'opponent_DEN']), -data['action_type_Finger Roll Shot']) + (data[
        'opponent_NYK'] - np.maximum(2.5, data['action_type_Tip Shot'] * 
        2.0))) / 2.0 * 2.0))) + np.tanh(data[
        'shot_zone_basic_In The Paint (Non-RA)'] * (data['playoffs_1'] > np
        .minimum(data['season_2010-11'] * data['minutes_remaining_2'] * 2.0,
        np.exp(-(data['opponent_DEN'] * (1.27586 - (data[
        'action_type_Driving Finger Roll Shot'] > data[
        'shot_zone_basic_In The Paint (Non-RA)']).astype(float)) * (data[
        'opponent_DEN'] * (1.27586 - (data[
        'action_type_Driving Finger Roll Shot'] > data[
        'shot_zone_basic_In The Paint (Non-RA)']).astype(float)))))) / 2.0)
        .astype(float)) + np.tanh(np.maximum(data[
        'action_type_Alley Oop Dunk Shot'], (np.maximum((np.maximum(data[
        'season_2010-11'], data['distance_21'] - np.tanh(np.minimum(
        6.788822650909424, data['opponent_MIL']))) > 0.147727).astype(float
        ) / 2.0 * 2.0, data['season_2003-04']) + (data['last_moments_0'] *
        ((data['period_4'] <= data['action_type_Driving Finger Roll Shot'])
        .astype(float) <= 6.0 + 1.23188 * 2.0).astype(float) + data[
        'action_type_Driving Finger Roll Shot'])) / 2.0)) + np.tanh(np.
        minimum(-data['shot_zone_area_Left Side(L)'] * data[
        'season_2011-12'], (data['distance_7'] <= (data[
        'minutes_remaining_6'] + np.tanh(np.minimum(data[
        'action_type_Turnaround Bank shot'], 0.594937))) / 2.0).astype(float))
        ) + np.tanh(data['action_type_Turnaround Jump Shot'] * data[
        'shot_zone_range_Less Than 8 ft.']) + np.tanh(data[
        'action_type_Jump Shot'] * ((0.147727 + np.minimum(-data[
        'shot_zone_area_Center(C)'], (0.886364 <= -((1.01449 > (data[
        'last_moments_1'] + 1.20833) / 2.0).astype(float) * 2.0 / 2.0) * (
        data['opponent_POR'] - data['opponent_GSW'])).astype(float))) / 2.0)
        ) + np.tanh(data['lat'] * np.maximum(data['season_2012-13'], (data[
        'opponent_SAS'] + np.tanh(data['shot_zone_range_Less Than 8 ft.'] *
        ((np.tanh((data['action_type_Driving Finger Roll Layup Shot'] > np.
        minimum(0.846154, -data[
        'action_type_Driving Finger Roll Layup Shot'])).astype(float)) / 
        2.0 - np.exp(-(np.tanh(data['opponent_UTA']) * 2.0 * (np.tanh(data[
        'opponent_UTA']) * 2.0)))) / 2.0))) / 2.0)) + np.tanh(data[
        'shot_zone_basic_Mid-Range'] * -np.maximum(data['opponent_MIL'],
        data['distance_13']) + data['action_type_Reverse Dunk Shot']
        ) + np.tanh(data['last_moments_1'] * (data[
        'shot_zone_area_Center(C)'] + (-data[
        'action_type_Driving Dunk Shot'] + 0.147727 * np.tanh(-(np.exp(-((
        data['playoffs_0'] - -((0.886364 + np.maximum(data[
        'minutes_remaining_8'], (data['opponent_DAL'] + np.exp(-(0.594937 *
        0.594937))) / 2.0)) / 2.0)) * (data['playoffs_0'] - -((0.886364 +
        np.maximum(data['minutes_remaining_8'], (data['opponent_DAL'] + np.
        exp(-(0.594937 * 0.594937))) / 2.0)) / 2.0)))) - data['distance_36'
        ] + 13.016348838806152))) / 2.0)) + np.tanh(data['opponent_IND'] *
        (data['distance_20'] - np.exp(-((1.0 > np.minimum(data[
        'opponent_OKC'], data['action_type_Pullup Jump shot'] + data[
        'distance_20'])).astype(float) * (1.0 > np.minimum(data[
        'opponent_OKC'], data['action_type_Pullup Jump shot'] + data[
        'distance_20'])).astype(float))))) + np.tanh(np.minimum(data[
        'season_2012-13'] * (data['opponent_DAL'] + data['opponent_POR']),
        data['action_type_Fadeaway Bank shot'])) + np.tanh(((-data[
        'action_type_Reverse Layup Shot'] + np.tanh((np.exp(-(0.0 * 0.0)) +
        np.minimum(0.147727, np.tanh(data['shot_zone_basic_Right Corner 3']
        ))) / 2.0)) / 2.0 + data['shot_zone_area_Left Side Center(LC)'] * (
        (data['distance_17'] - (0.472727 <= data['opponent_NYK']).astype(
        float)) / 2.0)) / 2.0) + np.tanh(data['lat'] * (np.maximum(data[
        'season_2013-14'] / 2.0 * 2.0, data['season_1998-99']) > np.minimum
        ((data['season_1998-99'] + 0.472727) / 2.0, np.minimum((3.33333 + (
        0.294118 - (data['action_type_Driving Finger Roll Layup Shot'] +
        data['season_2014-15'])) * 2.0) / 2.0, data[
        'action_type_Driving Slam Dunk Shot']) / 2.0)).astype(float)
        ) + np.tanh(data['action_type_Dunk'] + np.maximum(data[
        'action_type_Driving Dunk Shot'], data['distance_26'])) + np.tanh(
        data['opponent_CLE'] * -(data['season_2009-10'] - np.minimum(
        0.490566, -np.maximum(data[
        'action_type_Driving Finger Roll Layup Shot'], -((data['distance_1'
        ] + data['distance_33']) / 2.0) + data[
        'action_type_Driving Slam Dunk Shot']) - data[
        'action_type_Driving Finger Roll Layup Shot'] * ((data[
        'action_type_Driving Slam Dunk Shot'] / 2.0 + data['opponent_DAL']) /
        2.0)))) + np.tanh(data['minutes_remaining_9'] * ((data[
        'shot_zone_range_8-16 ft.'] + data['distance_26']) / 2.0)) + np.tanh(
        -(data['shot_zone_range_8-16 ft.'] - 0.472727) * (data[
        'season_2008-09'] > data['action_type_Running Hook Shot']).astype(
        float)) + np.tanh(data['season_2015-16'] * (data['opponent_HOU'] -
        data['action_type_Driving Dunk Shot'])) + np.tanh(np.maximum(data[
        'distance_24'] * data['period_3'], np.maximum(data[
        'action_type_Driving Finger Roll Shot'], data['opponent_PHX']) - 2.5)
        ) + np.tanh(data['distance_14'] * (data['minutes_remaining_11'] *
        np.tanh(np.exp(-((data['opponent_POR'] + (data[
        'action_type_Turnaround Bank shot'] - np.tanh(data[
        'action_type_Turnaround Bank shot'] / 2.0) * 2.0) * 2.0) * (data[
        'opponent_POR'] + (data['action_type_Turnaround Bank shot'] - np.
        tanh(data['action_type_Turnaround Bank shot'] / 2.0) * 2.0) * 2.0)))))
        ) + np.tanh(data['distance_25'] * ((data['away_False'] + np.minimum
        (math.tanh(0.472727), (((float(0.147727 > math.exp(-(0.886364 * 
        0.886364))) <= (data['opponent_ATL'] * -data[
        'action_type_Floating Jump shot'] > data['away_False']).astype(
        float)).astype(float) * 2.0 <= (data[
        'action_type_Floating Jump shot'] + np.tanh(0.472727)) / 2.0).
        astype(float) + data['season_2010-11']) / 2.0)) / 2.0 / 2.0) - data
        ['opponent_OKC']) + np.tanh(data['lon'] * ((data[
        'minutes_remaining_4'] + -np.minimum(data[
        'action_type_Driving Reverse Layup Shot'], -data['season_2003-04'])
        ) / 2.0)) + np.tanh(data['distance_27'] * (data[
        'shot_zone_basic_Backcourt'] + np.tanh(np.tanh(data[
        'shot_zone_area_Left Side Center(LC)']) - np.minimum((data[
        'season_1998-99'] + data['distance_3']) / 2.0, data[
        'action_type_Layup Shot']))) * ((1.80645 + 42.5 / 2.0) / 2.0)
        ) + np.tanh(data['season_2000-01'] * np.tanh(np.maximum(data[
        'action_type_Turnaround Jump Shot'], np.maximum(data['distance_13'],
        data['shot_zone_area_Right Side Center(RC)'] * (1.80645 / 2.0))))
        ) + np.tanh(np.tanh((np.minimum(data['opponent_SAC'], np.minimum(
        0.886364 * 2.0, np.tanh(data['opponent_SAC'] * data['distance_18'])
        )) + (data['shot_zone_area_Left Side Center(LC)'] + data[
        'action_type_Pullup Jump shot'] * np.exp(-((data[
        'shot_zone_area_Left Side Center(LC)'] <= np.minimum(data[
        'shot_zone_area_Left Side Center(LC)'] * 2.0, np.tanh(np.exp(-(np.
        exp(-(8.0 * 8.0)) * np.exp(-(8.0 * 8.0))))))).astype(float) * (data
        ['shot_zone_area_Left Side Center(LC)'] <= np.minimum(data[
        'shot_zone_area_Left Side Center(LC)'] * 2.0, np.tanh(np.exp(-(np.
        exp(-(8.0 * 8.0)) * np.exp(-(8.0 * 8.0))))))).astype(float))))) / 2.0)
        ) + np.tanh(-np.tanh(np.tanh(data['action_type_Dunk Shot'])) - data
        ['shot_zone_basic_Above the Break 3'] * ((data['opponent_DAL'] >
        data['minutes_remaining_7']).astype(float) <= data['period_5'] * -(
        data['action_type_Fadeaway Bank shot'] > (3.0 > data['distance_38'] -
        np.tanh((-np.exp(-(data['action_type_Slam Dunk Shot'] * data[
        'action_type_Slam Dunk Shot'])) + np.minimum(data[
        'shot_zone_basic_Left Corner 3'], np.tanh(data[
        'minutes_remaining_7']))) / 2.0)).astype(float)).astype(float)).
        astype(float))
    return Outputs(0.1 * predictions)


def GPIndividual2(data):
    predictions = np.tanh(5.0 * (data['distance_1'] + (-3.0 + np.maximum(
        0.365854, (0.540541 + 1.21875 * (data[
        'action_type_Pullup Jump shot'] + (0.365854 + data[
        'action_type_Dunk Shot']))) / 2.0)) / 2.0)) + np.tanh(data[
        'action_type_Fadeaway Jump Shot'] - np.maximum(-1.0, data[
        'distance_8'] * data['action_type_Dunk Shot'] + np.maximum(data[
        'shot_zone_range_24+ ft.'], np.minimum((data['minutes_remaining_0'] +
        data['distance_16']) / 2.0, 2.83333)))) + np.tanh(7.724756717681885 *
        (data['shot_zone_basic_Restricted Area'] * ((np.maximum(1.18667, -(
        -3.0 - data['shot_zone_basic_Restricted Area'])) + (2.0 + (data[
        'shot_zone_basic_Above the Break 3'] <= 1.570796).astype(float)) / 
        2.0) / 2.0)) - -data['action_type_Step Back Jump shot']) + np.tanh(
        2.718282 * ((6.057895660400391 * np.maximum(np.maximum(data[
        'shot_zone_range_Less Than 8 ft.'], (0.235294 <= data[
        'action_type_Step Back Jump shot']).astype(float)) - 0.365854, data
        ['action_type_Running Jump Shot'] - 2.68182) + np.tanh(data[
        'shot_zone_range_Less Than 8 ft.'])) / 2.0)) + np.tanh(-3.0 * (data
        ['action_type_Jump Shot'] * -((data['action_type_Hook Shot'] + (-
        3.0 - 3.141593)) / 2.0))) + np.tanh(9.0 * (((data['last_moments_0'] +
        np.tanh(data['distance_37'])) / 2.0 + np.minimum(data['distance_8'] -
        (data['action_type_Driving Layup Shot'] > 0.0).astype(float), data[
        'action_type_Driving Layup Shot'])) / 2.0 - (data[
        'action_type_Jump Shot'] - data['action_type_Driving Layup Shot']))
        ) + np.tanh(5.882203578948975 * ((np.tanh(data['distance_24'] - (
        data['action_type_Jump Shot'] + data['action_type_Tip Shot'])) -
        data['action_type_Jump Shot'] + -data['action_type_Tip Shot']) / 2.0)
        ) + np.tanh((data['distance_21'] + data[
        'action_type_Slam Dunk Shot']) / 2.0 + np.maximum(data[
        'distance_15'], (data['action_type_Slam Dunk Shot'] + (data[
        'action_type_Slam Dunk Shot'] <= (data['shot_zone_basic_Mid-Range'] +
        -0.367879) / 2.0).astype(float)) / 2.0) + (data[
        'action_type_Slam Dunk Shot'] + (data[
        'action_type_Alley Oop Layup shot'] > data[
        'shot_zone_range_8-16 ft.'] * (data['shot_zone_basic_Mid-Range'] -
        (data['minutes_remaining_9'] + data['opponent_MEM']))).astype(float))
        ) + np.tanh(np.tanh(data['minutes_remaining_9']) - (data[
        'shot_zone_area_Left Side(L)'] + -(0.365854 * (data[
        'action_type_Alley Oop Dunk Shot'] - data['season_2011-12'])) +
        data['season_1996-97']) / 2.0 - data['minutes_remaining_0']) + np.tanh(
        -3.0 * (0.711864 + (data['action_type_Jump Shot'] - -np.minimum(
        data['distance_25'] * (data['season_2015-16'] - 3.0 * 1.6), data[
        'action_type_Jump Shot'] + data['season_2006-07'] * ((data[
        'action_type_Jump Shot'] + np.minimum(1.6, data['distance_1'] *
        data['action_type_Jump Shot'])) / 2.0))))) + np.tanh(data[
        'action_type_Driving Layup Shot'] + (data[
        'shot_zone_range_16-24 ft.'] + data['shot_zone_range_8-16 ft.']) + 
        --np.maximum(np.maximum(data['distance_21'], data[
        'shot_zone_range_8-16 ft.'] - (-1.0 + 2.718282)), data[
        'action_type_Slam Dunk Shot'] - -(0.367879 <= data[
        'action_type_Driving Layup Shot']).astype(float))) + np.tanh((data[
        'period_1'] + np.maximum(12.768333435058594 * ((np.maximum(data[
        'action_type_Dunk'], np.maximum(np.maximum(data[
        'action_type_Running Jump Shot'], (data['away_False'] + data[
        'distance_26']) / 2.0 + data['action_type_Finger Roll Layup Shot']),
        data['season_2008-09'])) + data['opponent_VAN'] + (data[
        'action_type_Driving Dunk Shot'] - -data[
        'action_type_Running Jump Shot'])) / 2.0) + data[
        'action_type_Running Jump Shot'], np.maximum(data['season_2008-09'],
        -data['minutes_remaining_11']))) / 2.0) + np.tanh(data['distance_0'
        ] - np.minimum(data['away_True'], (data[
        'action_type_Turnaround Fadeaway shot'] + np.maximum(0.571429 * ((
        data['last_moments_1'] + -1.34211) / 2.0) - data['opponent_BKN'], -
        data['distance_6']) > 1.21875).astype(float)) - data[
        'last_moments_1'] + data['distance_24']) + np.tanh(data[
        'action_type_Running Hook Shot'] + np.tanh(data['distance_23']) *
        data['distance_23'] + -(data['period_4'] * np.tanh(np.maximum(data[
        'distance_12'], data['opponent_VAN'])) <= data['period_4']).astype(
        float) + data['distance_22']) + np.tanh((data[
        'action_type_Running Jump Shot'] > (0.076923 > 0.367879 - np.
        minimum((0.06383 > data['minutes_remaining_3']).astype(float), data
        ['season_1996-97'] * 1.0)).astype(float)).astype(float) - (data[
        'action_type_Layup Shot'] - ((data['action_type_Jump Bank Shot'] + 
        1.34211) / 2.0 + -(data['season_2015-16'] + data[
        'action_type_Layup Shot'])))) + np.tanh((data[
        'action_type_Pullup Jump shot'] + data[
        'action_type_Driving Finger Roll Shot']) / 2.0 - np.minimum(np.tanh
        (np.tanh(2.68182)), 3.141593) - (1.18667 - np.maximum(data[
        'action_type_Jump Bank Shot'], data['action_type_Running Jump Shot']))
        ) + np.tanh((np.maximum((data['action_type_Jump Bank Shot'] + np.
        tanh((0.210526 > data['shot_zone_area_Left Side(L)'] - np.minimum(
        data['season_2005-06'], data['action_type_Turnaround Fadeaway shot'
        ])).astype(float))) / 2.0, data['season_2005-06']) + 0.142857) / 
        2.0 + (data['action_type_Slam Dunk Shot'] - (data[
        'shot_zone_basic_Backcourt'] + -data['shot_zone_area_Right Side(R)'
        ]) * -data['playoffs_0'])) + np.tanh(0.457627 - data[
        'action_type_Layup Shot'] - (data['action_type_Layup Shot'] - (
        0.142857 - np.maximum(data['shot_zone_area_Left Side(L)'], data[
        'action_type_Layup Shot'] + data['action_type_Tip Shot'])))) + np.tanh(
        data['opponent_SAC'] + (data['opponent_PHX'] + (data[
        'season_2000-01'] + (data['season_1998-99'] + (data['distance_25'] -
        (data['lon'] + data['season_2011-12'])))) / 2.0)) + np.tanh((np.
        tanh(data['opponent_UTA']) + data['action_type_Driving Dunk Shot'] +
        (data['action_type_Slam Dunk Shot'] + (data['last_moments_0'] +
        data['action_type_Driving Layup Shot'])) / 2.0 + (data[
        'action_type_Driving Layup Shot'] - ((0.365854 + -data[
        'action_type_Slam Dunk Shot'] > data['last_moments_0']).astype(
        float) <= data['distance_0']).astype(float))) / 2.0) + np.tanh(-(
        data['action_type_Layup Shot'] - (np.minimum((-2.718282 <= (data[
        'distance_34'] <= 3.141593).astype(float)).astype(float), 2.0) *
        data['last_moments_0'] - (data['shot_zone_basic_Mid-Range'] + (data
        ['action_type_Layup Shot'] + data['action_type_Layup Shot'])) / 2.0))
        ) + np.tanh(data['action_type_Alley Oop Dunk Shot'] + (data[
        'action_type_Dunk Shot'] + (data['distance_1'] + data['distance_1'] -
        (data['distance_35'] + -np.minimum(data['distance_19'], 0.076923) *
        (1.34211 + np.minimum(0.142857, data['distance_18']))) - data[
        'season_2014-15'])) / 2.0) + np.tanh(data[
        'action_type_Driving Dunk Shot'] + (np.maximum((data[
        'action_type_Pullup Jump shot'] + data['action_type_Jump Bank Shot'
        ]) / 2.0, data['season_1999-00'] - (0.571429 <= np.tanh(np.minimum(
        0.210526, data['opponent_MEM']))).astype(float)) + np.tanh((data[
        'action_type_Jump Bank Shot'] + (data[
        'action_type_Driving Dunk Shot'] + (data[
        'action_type_Floating Jump shot'] + data[
        'action_type_Fadeaway Jump Shot'])) / 2.0) / 2.0))) + np.tanh(-(
        data['action_type_Jump Shot'] - (data['minutes_remaining_5'] - (
        data['action_type_Jump Shot'] - data[
        'shot_zone_basic_Left Corner 3'] * 0.586957)))) + np.tanh(np.
        maximum(data['season_2007-08'], (data['shot_zone_range_16-24 ft.'] +
        np.maximum(np.minimum(data['distance_34'], np.tanh(data[
        'opponent_SAS'])), np.maximum(data['distance_23'] - data[
        'distance_42'], np.maximum(2.718282 * data[
        'shot_zone_area_Right Side Center(RC)'], data[
        'action_type_Jump Bank Shot'])))) / 2.0)) + np.tanh(np.maximum(data
        ['action_type_Running Bank shot'], np.maximum(data[
        'action_type_Running Jump Shot'], np.maximum(data[
        'action_type_Running Hook Shot'] - (data['period_5'] + (0.142857 > 
        data['shot_zone_area_Back Court(BC)'] - 1.0).astype(float)) / 2.0, 
        np.maximum((data['action_type_Dunk'] + 0.31831) / 2.0, data[
        'action_type_Floating Jump shot']) + (data[
        'action_type_Running Jump Shot'] + data['away_True']) / 2.0)) + np.
        tanh(-1.0))) + np.tanh((data['shot_zone_area_Back Court(BC)'] *
        data['action_type_Driving Layup Shot'] + 0.235294) / 2.0 - (data[
        'action_type_Fadeaway Jump Shot'] + ((data[
        'shot_zone_range_Less Than 8 ft.'] + data['action_type_Jump Shot']) /
        2.0 - data['action_type_Driving Layup Shot']))) + np.tanh((data[
        'distance_21'] + (-data['action_type_Jump Shot'] + 0.0)) / 2.0 + -(
        data['season_2015-16'] - (data['shot_zone_basic_Right Corner 3'] >
        (-data['action_type_Turnaround Bank shot'] * (data['distance_38'] >
        -1.34211 + (data['action_type_Running Bank shot'] - data[
        'shot_zone_basic_Right Corner 3'])).astype(float) <= 1.570796 *
        data['action_type_Layup']).astype(float)).astype(float))) + np.tanh(
        data['action_type_Slam Dunk Shot'] + --((data[
        'shot_zone_range_16-24 ft.'] + ((data['distance_22'] > 0.210526).
        astype(float) + np.maximum(data['shot_zone_area_Left Side(L)'],
        data['shot_zone_area_Right Side Center(RC)']))) / 2.0)) + np.tanh(
        data['shot_zone_range_24+ ft.'] - data['action_type_Jump Shot'] * (
        data['shot_zone_basic_Mid-Range'] * ((data[
        'shot_zone_range_Less Than 8 ft.'] + data[
        'action_type_Alley Oop Dunk Shot'] * data['opponent_MIA']) / 2.0 *
        data['action_type_Driving Dunk Shot']) - (data['away_True'] - data[
        'action_type_Driving Dunk Shot']))) + np.tanh((data[
        'action_type_Dunk'] + (data['action_type_Slam Dunk Shot'] + -(data[
        'distance_41'] > data['action_type_Slam Dunk Shot']).astype(float) *
        (0.210526 + (data['action_type_Fadeaway Bank shot'] <= np.tanh(data
        ['distance_27'])).astype(float))) / 2.0) / 2.0 - (data[
        'shot_zone_basic_Restricted Area'] + (data['action_type_Jump Shot'] +
        (0.586957 - (data['action_type_Slam Dunk Shot'] - np.tanh(data[
        'distance_5'])))) / 2.0)) + np.tanh((-(data['season_2014-15'] + (
        data['action_type_Fadeaway Jump Shot'] + -(data['last_moments_0'] *
        np.tanh(np.tanh(10.0 + np.maximum(0.457627, (data['season_2008-09'] <=
        np.minimum(data['away_False'], 0.63662 + np.minimum(2.0, 0.210526))
        ).astype(float))) + -data['distance_17'])))) + data[
        'last_moments_0']) / 2.0) + np.tanh(data[
        'action_type_Alley Oop Dunk Shot'] + data['distance_0'] * data[
        'away_True']) + np.tanh((np.minimum(data['period_5'], data[
        'last_moments_0']) + (np.maximum(data[
        'action_type_Driving Dunk Shot'], data[
        'action_type_Running Bank shot']) + -(data['away_False'] + np.tanh(
        data['distance_6']))) / 2.0) / 2.0) + np.tanh((data[
        'shot_zone_range_8-16 ft.'] > 0.0).astype(float) + (data[
        'action_type_Driving Finger Roll Layup Shot'] + -(data[
        'opponent_OKC'] - (data['action_type_Driving Slam Dunk Shot'] + -(
        data['opponent_OKC'] - data['action_type_Driving Slam Dunk Shot'])) /
        2.0)) / 2.0) + np.tanh((data[
        'action_type_Driving Reverse Layup Shot'] + data[
        'action_type_Slam Dunk Shot']) * (data['distance_2'] + (1.18667 -
        np.maximum(data['opponent_NYK'], data[
        'action_type_Finger Roll Layup Shot']))) + np.minimum(3.141593, np.
        maximum(data['action_type_Reverse Dunk Shot'], np.minimum(np.
        maximum(data['distance_22'], 0.586957), np.maximum(data[
        'action_type_Driving Layup Shot'], data['distance_2']))))) + np.tanh(
        data['action_type_Jump Shot'] * np.minimum(data[
        'action_type_Fadeaway Jump Shot'], np.minimum((data['distance_5'] +
        (0.365854 + np.tanh(data['shot_zone_basic_Restricted Area']) * data
        ['opponent_CHA'])) / 2.0, -data[
        'shot_zone_basic_In The Paint (Non-RA)']))) + np.tanh(-((data[
        'season_1997-98'] + np.tanh(data['minutes_remaining_8'])) / 2.0 + 
        data['minutes_remaining_0'] * ((data['last_moments_1'] + data[
        'opponent_DAL']) / 2.0))) + np.tanh(((data['distance_19'] + (data[
        'action_type_Driving Finger Roll Shot'] - data['season_2015-16'])) /
        2.0 + (data['action_type_Driving Slam Dunk Shot'] * (0.142857 > np.
        tanh(np.maximum(data['shot_zone_basic_In The Paint (Non-RA)'], -
        data['action_type_Running Hook Shot']))).astype(float) + data[
        'action_type_Running Hook Shot']) / 2.0 * (1.570796 + data[
        'distance_19'])) / 2.0) + np.tanh(0.235294 - (data[
        'action_type_Layup Shot'] + (data['shot_zone_basic_Mid-Range'] +
        data['action_type_Tip Shot']) / 2.0)) + np.tanh(data[
        'action_type_Alley Oop Dunk Shot'] + np.minimum(data[
        'season_2005-06'], 1.18667 - (data['shot_zone_range_24+ ft.'] +
        data['action_type_Hook Shot']) + data[
        'action_type_Alley Oop Dunk Shot'])) + np.tanh(data['period_4'] *
        data['distance_17'] + np.maximum(data['distance_34'], np.tanh(
        0.210526)) + np.maximum(data[
        'action_type_Driving Finger Roll Layup Shot'], data[
        'action_type_Driving Finger Roll Layup Shot'] - -(data[
        'distance_14'] > data['distance_34']).astype(float))) + np.tanh(np.
        maximum(data['action_type_Driving Finger Roll Shot'] + (data[
        'shot_zone_basic_Restricted Area'] - np.minimum(1.42553, np.maximum
        (data['action_type_Alley Oop Layup shot'], 3.141593))) * (data[
        'season_2002-03'] * data['opponent_SEA']), np.tanh(data[
        'action_type_Reverse Dunk Shot']))) + np.tanh(-(data[
        'action_type_Layup Shot'] + ((data['minutes_remaining_0'] + 
        0.711864 <= data['shot_zone_basic_Mid-Range']).astype(float) + np.
        minimum(data['shot_zone_basic_Above the Break 3'], (data[
        'shot_zone_area_Right Side(R)'] + data['opponent_DAL']) * ((data[
        'action_type_Layup Shot'] + -(np.minimum(data['opponent_NOP'], 
        0.31831) > 0.571429 + 1.570796).astype(float)) / 2.0))))) + np.tanh(np
        .minimum((-((data['action_type_Step Back Jump shot'] + (0.076923 <=
        (-3.0 + (0.06383 > data['action_type_Layup Shot']).astype(float)) /
        2.0).astype(float)) / 2.0) + (data['action_type_Layup Shot'] + data
        ['season_2005-06']) / 2.0) / 2.0 - data[
        'shot_zone_basic_In The Paint (Non-RA)'], -data['distance_25'] -
        data['action_type_Layup Shot'])) + np.tanh((data[
        'action_type_Reverse Dunk Shot'] - data['action_type_Layup Shot'] +
        data['shot_zone_area_Center(C)']) / 2.0 + data[
        'action_type_Slam Dunk Shot']) + np.tanh((data[
        'action_type_Slam Dunk Shot'] + (np.maximum(data['opponent_NOH'],
        np.maximum(data['season_2006-07'], np.maximum(data[
        'action_type_Driving Dunk Shot'], (np.maximum(data['distance_23'], 
        (data['opponent_CHI'] + 0.235294) / 2.0) + np.tanh(data[
        'action_type_Jump Bank Shot'])) / 2.0))) + np.tanh(-np.tanh((data[
        'action_type_Driving Dunk Shot'] - (2.0 <= data[
        'minutes_remaining_6']).astype(float) > data['distance_12']).astype
        (float))))) / 2.0) + np.tanh(np.maximum(data['distance_1'], (np.
        maximum(data['minutes_remaining_7'], (data['action_type_Tip Shot'] <=
        data['distance_0'] - np.maximum((np.minimum(--1.0, data[
        'action_type_Running Jump Shot']) > 2.68182).astype(float), np.
        minimum(0.210526, np.maximum(1.42553, data['minutes_remaining_1'])) -
        0.31831)).astype(float)) > -np.maximum(data['minutes_remaining_7'],
        data['distance_25'])).astype(float))) + np.tanh((data[
        'action_type_Alley Oop Dunk Shot'] + (0.076923 + data[
        'action_type_Jump Shot']) * (data['distance_15'] + np.maximum(data[
        'distance_41'], data['opponent_POR'] - 1.0))) / 2.0) + np.tanh(data
        ['action_type_Slam Dunk Shot'] - np.minimum(np.minimum(2.68182, 
        data['last_moments_0'] + data['action_type_Reverse Dunk Shot']),
        data['action_type_Slam Dunk Shot']) * ((data[
        'action_type_Driving Slam Dunk Shot'] + (data[
        'shot_type_2PT Field Goal'] + (data['action_type_Slam Dunk Shot'] +
        -np.tanh(data['action_type_Slam Dunk Shot'])) / 2.0)) / 2.0)
        ) + np.tanh(data['away_True'] * ((data['season_2009-10'] + 
        7.875600814819336 * -np.maximum(data['distance_42'], data[
        'action_type_Fadeaway Jump Shot'] + data['season_2012-13'] * data[
        'action_type_Driving Finger Roll Shot'])) / 2.0)) + np.tanh(np.
        minimum(float(max(0.235294, 1.570796) > 0.076923), np.tanh(((
        0.076923 + data['lat'] <= (data['opponent_UTA'] * 0.711864 > -(data
        ['action_type_Pullup Jump shot'] * np.tanh(np.maximum(0.90625, 
        0.367879)))).astype(float)).astype(float) > -np.minimum(data[
        'action_type_Running Bank shot'], data['lat'])).astype(float)))
        ) + np.tanh(data['action_type_Layup Shot'] * np.tanh(data[
        'season_1998-99']) - (data['action_type_Layup Shot'] > data[
        'opponent_CHA']).astype(float)) + np.tanh(-(((data['distance_10'] +
        data['last_moments_1']) / 2.0 + data[
        'action_type_Turnaround Jump Shot'] * (data[
        'action_type_Slam Dunk Shot'] - 9.907742500305176 * ((data[
        'season_2011-12'] + -1.0 * data['action_type_Turnaround Jump Shot']
        ) / 2.0))) / 2.0 * (2.0 * np.maximum((min(-2.0, 0.0) > data[
        'minutes_remaining_8']).astype(float), data[
        'action_type_Slam Dunk Shot']) - np.tanh(data['season_2000-01'])))
        ) + np.tanh((data['shot_zone_basic_Mid-Range'] * (data[
        'season_1999-00'] - np.tanh(data['opponent_TOR'] * np.tanh(np.
        minimum(data['action_type_Bank Shot'], -data[
        'action_type_Hook Shot'])))) + data['distance_3'] * (1.18667 * ((-
        3.0 + ((0.365854 > data['distance_1']).astype(float) <= data[
        'opponent_SEA'] * np.tanh((data['distance_40'] > data[
        'action_type_Hook Shot']).astype(float))).astype(float)) / 2.0))) / 2.0
        ) + np.tanh(data['shot_zone_basic_Above the Break 3'] * ((data[
        'season_1997-98'] + -(((0.586957 > data['opponent_WAS']).astype(
        float) + np.maximum(np.tanh(np.maximum(data['distance_1'], data[
        'distance_1'])), (data['distance_1'] + data['opponent_NYK']) / 2.0)
        ) / 2.0)) / 2.0)) + np.tanh((np.maximum(data[
        'action_type_Jump Bank Shot'], ((data['action_type_Dunk Shot'] <= 
        0.711864).astype(float) <= data['action_type_Jump Bank Shot']).
        astype(float)) + data['away_True'] * (data['lat'] * (2.0 + data[
        'period_3']))) / 2.0) + np.tanh(np.minimum(-data[
        'action_type_Reverse Layup Shot'], -(data['season_2015-16'] - -(
        data['action_type_Driving Layup Shot'] + data[
        'shot_zone_basic_Backcourt'])))) + np.tanh(2.0 * np.tanh(np.tanh(np
        .tanh(--(data['action_type_Layup Shot'] * ((data['season_2012-13'] +
        (data['lat'] <= np.minimum(0.90625, data[
        'action_type_Reverse Dunk Shot'] + (0.0 > (data['opponent_MIA'] + 
        1.18667 * -(data['distance_45'] > np.tanh(0.142857)).astype(float)) /
        2.0).astype(float))).astype(float)) / 2.0)))))) + np.tanh(data[
        'distance_7'] * data['minutes_remaining_2'] - (data[
        'action_type_Tip Shot'] + np.minimum(data['opponent_SAC'], (data[
        'opponent_CLE'] + (np.minimum(2.0, 0.586957) * float(0.0 <= 1.18667
        ) + (data['opponent_WAS'] > 1.570796).astype(float)) / 2.0) / 2.0)) /
        2.0) + np.tanh(-((data['minutes_remaining_4'] + (data[
        'action_type_Jump Shot'] + np.maximum(data['opponent_VAN'], data[
        'action_type_Driving Layup Shot'])) / 2.0) * (data[
        'shot_type_2PT Field Goal'] - (data['period_7'] - np.minimum((data[
        'action_type_Slam Dunk Shot'] + 2.68182) / 2.0 * data[
        'action_type_Driving Reverse Layup Shot'] + 0.711864, (data[
        'season_2013-14'] > data['minutes_remaining_6'] - (data[
        'shot_zone_range_Less Than 8 ft.'] + 0.571429) / 2.0).astype(float)))))
        ) + np.tanh(data['opponent_PHI'] * (data[
        'action_type_Running Jump Shot'] + (data[
        'action_type_Driving Layup Shot'] - (1.21875 + data['opponent_VAN']
        ) / 2.0))) + np.tanh(data['action_type_Alley Oop Dunk Shot'] + np.
        minimum((0.31831 + np.minimum(data['shot_type_3PT Field Goal'], (
        data['opponent_NOP'] > -1.0).astype(float))) / 2.0, -(data[
        'season_2014-15'] - ((np.minimum(np.maximum(data['season_2012-13'],
        1.6), 0.235294) > data['minutes_remaining_1'] + data[
        'action_type_Jump Bank Shot']).astype(float) + --(data[
        'distance_35'] - data['opponent_BKN']))))) + np.tanh(data[
        'action_type_Jump Shot'] * data['opponent_DEN'] - np.tanh((-(data[
        'action_type_Driving Finger Roll Shot'] - data[
        'action_type_Layup Shot'] * np.maximum(data['last_moments_0'], np.
        tanh((data['action_type_Layup Shot'] + (4.15385 > data[
        'action_type_Running Layup Shot']).astype(float)) / 2.0))) + (data[
        'period_5'] + np.maximum(0.457627, 0.711864 + -1.0))) / 2.0)
        ) + np.tanh(data['away_False'] * (data[
        'action_type_Running Jump Shot'] * np.maximum(data['season_2000-01'
        ], (data['opponent_SAC'] + ((data['opponent_ATL'] > data[
        'distance_27']).astype(float) + (1.42553 + np.minimum((data[
        'action_type_Running Jump Shot'] > data[
        'action_type_Pullup Jump shot']).astype(float), (3.0 > (data[
        'opponent_DET'] + -0.31831) / 2.0).astype(float))) / 2.0)) / 2.0))
        ) + np.tanh((data['lat'] > np.maximum(0.90625, (0.06383 + data[
        'minutes_remaining_4'] * np.minimum(0.210526, data[
        'minutes_remaining_4'] + float(0.076923 <= 0.0)) + data[
        'season_2004-05']) / 2.0)).astype(float)) + np.tanh(((data[
        'minutes_remaining_6'] + data['season_2001-02']) / 2.0 + (data[
        'opponent_PHI'] + data['action_type_Driving Slam Dunk Shot']) / 2.0 +
        np.minimum(data['opponent_BOS'], np.maximum(data['distance_43'],
        data['distance_43']))) / 2.0) + np.tanh(-(data['opponent_NJN'] - (
        data['distance_38'] + ((data['distance_19'] + (-data['opponent_HOU'
        ] - data['action_type_Tip Shot'])) / 2.0 + (data[
        'action_type_Alley Oop Dunk Shot'] > (data['distance_29'] <= data[
        'action_type_Layup'] * (1.18667 * (data['distance_41'] - (2.83333 +
        5.020663738250732 + 0.31831) / 2.0))).astype(float)).astype(float)) /
        2.0))) + np.tanh(data['minutes_remaining_4'] * (np.tanh(np.tanh(
        data['opponent_NOP'] - np.tanh(data['minutes_remaining_1']) * data[
        'shot_zone_basic_Restricted Area'])) * (-3.0 - (1.6 + ((1.21875 - 
        10.551502227783203 <= np.tanh(2.0)).astype(float) - data[
        'minutes_remaining_4'] * 0.076923 * data['season_1998-99'])) / 2.0))
        ) + np.tanh(data['action_type_Dunk'] + (data[
        'action_type_Driving Finger Roll Shot'] * np.maximum(data[
        'distance_31'], 2.0) > np.minimum((data['opponent_UTA'] + (data[
        'minutes_remaining_7'] + (1.21875 + 0.076923) / 2.0) / 2.0) / 2.0, 
        -(data['distance_25'] + -1.18667)) * data['distance_20']).astype(float)
        ) + np.tanh((data['action_type_Fadeaway Bank shot'] + (np.minimum(
        data['opponent_LAC'], data['shot_zone_range_16-24 ft.']) + np.
        maximum(data['action_type_Jump Hook Shot'], data[
        'action_type_Driving Dunk Shot'] * 0.586957))) / 2.0) + np.tanh(
        data['away_False'] * (data['action_type_Driving Layup Shot'] + (np.
        maximum(data['season_2006-07'], data['opponent_GSW'] * np.tanh(np.
        minimum(0.235294, 2.68182))) + data['action_type_Reverse Dunk Shot']))
        ) + np.tanh(-(data['period_1'] * np.minimum(0.711864, np.maximum(
        data['distance_22'], data['action_type_Turnaround Fadeaway shot'] +
        (data['shot_zone_basic_Restricted Area'] + (data['opponent_DAL'] +
        np.minimum(2.718282, np.tanh(np.tanh(data['distance_23'] - 0.367879
        ))))) / 2.0)))) + np.tanh((data['action_type_Driving Dunk Shot'] +
        (data['away_False'] * -data['action_type_Jump Shot'] + data[
        'action_type_Slam Dunk Shot'])) / 2.0) + np.tanh(data[
        'action_type_Slam Dunk Shot'] + (np.maximum(0.90625, data['lat']) <=
        1.0 * data['away_False']).astype(float)) + np.tanh(data[
        'season_2011-12'] * ((data['shot_zone_basic_Restricted Area'] + 
        0.90625 * ((data['action_type_Jump Shot'] > data[
        'action_type_Alley Oop Dunk Shot']).astype(float) * (np.minimum(
        data['action_type_Alley Oop Dunk Shot'], 11.166929244995117) <=
        data['distance_39']).astype(float))) / 2.0)) + np.tanh((data[
        'minutes_remaining_2'] * (-data['opponent_GSW'] * (4.0 * ((data[
        'action_type_Layup Shot'] + np.minimum(--data[
        'action_type_Jump Bank Shot'], -np.tanh(data[
        'action_type_Driving Reverse Layup Shot']))) / 2.0))) + (0.06383 + 
        (data['action_type_Running Bank shot'] + np.tanh(np.minimum(data[
        'period_6'], data['shot_type_3PT Field Goal']))) / 2.0)) / 2.0
        ) + np.tanh(np.maximum(data['opponent_CHA'], (data[
        'action_type_Driving Dunk Shot'] + np.maximum(data['distance_1'], -
        (math.tanh(-0.63662) + -(data['period_3'] * data['distance_24'])))) /
        2.0)) + np.tanh(data['opponent_HOU'] * (data[
        'action_type_Jump Shot'] + (data['opponent_BOS'] + np.minimum(data[
        'minutes_remaining_8'] + np.maximum(2.0, np.minimum(data[
        'opponent_NOH'], 0.90625)), -data['distance_23'])))) + np.tanh((-
        data['action_type_Tip Shot'] + (data[
        'action_type_Alley Oop Dunk Shot'] - (data['distance_10'] + (
        0.90625 + (data['action_type_Alley Oop Dunk Shot'] - data[
        'opponent_NYK']) * np.minimum(3.0, data['opponent_NYK']))) / 2.0)) /
        2.0) + np.tanh(data['action_type_Alley Oop Dunk Shot'] * (data[
        'distance_40'] + (math.tanh(2.0) <= data[
        'action_type_Alley Oop Dunk Shot']).astype(float)) - (data[
        'action_type_Hook Shot'] + data['action_type_Tip Shot'])) + np.tanh(
        data['action_type_Slam Dunk Shot'] + (data['away_False'] * data[
        'action_type_Running Jump Shot'] + np.maximum(data[
        'action_type_Running Bank shot'], data[
        'action_type_Turnaround Bank shot'] * (0.142857 - np.minimum(data[
        'action_type_Turnaround Bank shot'] * -2.0, data[
        'action_type_Running Bank shot'])))) / 2.0) + np.tanh(data[
        'action_type_Dunk'] + data['season_2010-11'] * np.maximum(data[
        'shot_zone_basic_In The Paint (Non-RA)'], np.minimum(data[
        'minutes_remaining_3'], np.maximum(data[
        'action_type_Driving Finger Roll Layup Shot'], data[
        'shot_zone_area_Center(C)'])) - np.maximum(0.540541, np.maximum(
        data['distance_30'], (data['minutes_remaining_9'] > 0.367879).
        astype(float))))) + np.tanh(data['action_type_Layup Shot'] * np.
        tanh((data['distance_2'] - 2.83333) * ((-data[
        'minutes_remaining_10'] + data['shot_zone_area_Back Court(BC)'] +
        np.maximum(np.minimum(0.63662, 0.63662), ---np.minimum(data[
        'shot_zone_area_Back Court(BC)'], np.maximum(4.15385, np.maximum(
        0.0, 0.457627))))) / 2.0))) + np.tanh((np.maximum(data[
        'action_type_Reverse Layup Shot'], data[
        'action_type_Driving Layup Shot'] * 1.34211) + (data[
        'shot_zone_area_Back Court(BC)'] - data['action_type_Dunk'] - -data
        ['distance_23'])) / 2.0 * data['shot_zone_range_8-16 ft.']) + np.tanh(
        -(data['action_type_Layup Shot'] * ((data['away_False'] + np.
        maximum(data['action_type_Alley Oop Dunk Shot'], data[
        'opponent_DET'])) / 2.0))) + np.tanh((data['opponent_GSW'] - (data[
        'action_type_Slam Dunk Shot'] - data['lon']) - (2.83333 + data[
        'action_type_Driving Finger Roll Layup Shot']) / 2.0) * data[
        'distance_5'] + ((0.210526 <= np.maximum(data[
        'action_type_Slam Dunk Shot'], data[
        'shot_zone_basic_Restricted Area'])).astype(float) <= data[
        'action_type_Driving Finger Roll Shot']).astype(float)) + np.tanh((
        data['action_type_Driving Dunk Shot'] + np.minimum(data[
        'action_type_Pullup Jump shot'], data['shot_zone_range_8-16 ft.'] *
        ((2.83333 * --1.0 + data['action_type_Driving Dunk Shot']) / 2.0 - 
        -data['action_type_Pullup Jump shot']))) / 2.0) + np.tanh((data[
        'opponent_DEN'] > data['action_type_Jump Shot'] * (np.tanh(0.367879
        ) + -(-(4.15385 <= data['distance_43']).astype(float) - (data[
        'last_moments_0'] + 0.0)))).astype(float) * np.tanh(data[
        'shot_zone_area_Center(C)'])) + np.tanh(data['last_moments_0'] * -(
        data['shot_zone_basic_Restricted Area'] + (data[
        'minutes_remaining_5'] + data['action_type_Finger Roll Shot']) / 2.0)
        ) + np.tanh(data['shot_type_3PT Field Goal'] * (data['opponent_MIL'
        ] + (data['season_2002-03'] + (0.540541 - (data['opponent_DAL'] >
        np.minimum(data['distance_28'] + 0.365854 * (data[
        'action_type_Turnaround Bank shot'] * (data[
        'action_type_Turnaround Bank shot'] <= 0.235294).astype(float)), -
        np.minimum(data['action_type_Turnaround Bank shot'], 2.83333))).
        astype(float))) / 2.0)) + np.tanh(data[
        'action_type_Driving Dunk Shot'] - (data['action_type_Hook Shot'] -
        np.maximum(data['distance_37'], ((data['opponent_MIL'] + np.maximum
        (data['action_type_Driving Dunk Shot'], 0.90625)) / 2.0 + (data[
        'action_type_Slam Dunk Shot'] - data['opponent_BKN'] - data[
        'action_type_Running Bank shot'])) / 2.0))) + np.tanh(-((data[
        'action_type_Tip Shot'] + np.maximum(np.tanh(3.0 * -np.tanh(np.tanh
        (data['lat']))) * (data['minutes_remaining_2'] + np.minimum(2.68182,
        data['distance_31'])) - data['action_type_Driving Slam Dunk Shot'],
        -(data['opponent_IND'] + 0.90625 * 0.7785810828208923))) / 2.0)
        ) + np.tanh(data['action_type_Driving Layup Shot'] * ((-data[
        'season_2000-01'] + ((3.141593 <= np.tanh(-(data[
        'minutes_remaining_4'] <= float(0.586957 <= 0.06383)).astype(float)
        )).astype(float) + np.maximum(data['action_type_Tip Shot'], data[
        'season_2006-07'] + np.tanh(np.tanh(float(1.18667 <= 0.711864))) +
        data['season_2014-15'])) / 2.0) / 2.0)) + np.tanh(data[
        'minutes_remaining_2'] * ((data['opponent_WAS'] + np.minimum(data[
        'opponent_CLE'], -(0.711864 * -12.80661392211914))) / 2.0)) + np.tanh(
        (data['action_type_Slam Dunk Shot'] + data[
        'shot_zone_area_Center(C)'] * (data['minutes_remaining_9'] * ((data
        ['distance_1'] + -(data['action_type_Driving Jump shot'] - -(
        0.540541 > -np.minimum(data['action_type_Running Layup Shot'], np.
        minimum((0.06383 > -data['season_2010-11']).astype(float), data[
        'distance_25']))).astype(float))) / 2.0))) / 2.0) + np.tanh(-((
        7.217765808105469 + data['action_type_Reverse Dunk Shot'] + (data[
        'lat'] + data['distance_22'] * np.minimum(data[
        'shot_zone_range_Less Than 8 ft.'], data['minutes_remaining_2'] * -
        -1.0))) / 2.0 * ((data['action_type_Hook Shot'] + (data[
        'action_type_Step Back Jump shot'] + np.minimum(0.457627, -np.
        maximum(data['action_type_Driving Finger Roll Layup Shot'], data[
        'minutes_remaining_2']))) / 2.0) / 2.0))) + np.tanh(data[
        'action_type_Fadeaway Bank shot'] + data['season_2011-12'] * np.
        maximum(data['opponent_NJN'], data['action_type_Layup Shot'] - np.
        minimum(4.15385 * 0.142857, (data['opponent_IND'] <= data[
        'action_type_Jump Bank Shot']).astype(float)))) + np.tanh(data[
        'action_type_Layup Shot'] * ((data['opponent_MIL'] + ((np.tanh(data
        ['action_type_Layup Shot']) * data['action_type_Fadeaway Bank shot'
        ] + data['season_2011-12']) / 2.0 + (0.711864 + np.tanh(data[
        'distance_19']))) / 2.0) / 2.0)) + np.tanh(np.tanh(data[
        'action_type_Finger Roll Layup Shot']) - data[
        'shot_zone_basic_Restricted Area'] * np.maximum(data['opponent_DEN'
        ], np.tanh(np.tanh(data['season_1997-98'])) - 0.457627 * -data[
        'action_type_Tip Shot'] * np.tanh(0.63662))) + np.tanh((data[
        'action_type_Running Hook Shot'] + np.maximum(np.maximum(data[
        'action_type_Alley Oop Dunk Shot'], data[
        'action_type_Driving Dunk Shot']), (data[
        'action_type_Alley Oop Dunk Shot'] + ((data[
        'action_type_Driving Dunk Shot'] > np.minimum(np.minimum(0.90625, (
        data['opponent_WAS'] <= -data['action_type_Driving Dunk Shot']).
        astype(float)), data['distance_40']) * data['period_1']).astype(
        float) + data['action_type_Driving Slam Dunk Shot'])) / 2.0)) / 2.0 +
        data['season_1996-97']) + np.tanh(np.maximum(data[
        'action_type_Slam Dunk Shot'], ((data[
        'action_type_Turnaround Bank shot'] + data['opponent_WAS']) / 2.0 +
        np.minimum((data['distance_29'] <= -data['action_type_Dunk']).
        astype(float), np.tanh((-data['action_type_Pullup Jump shot'] *
        data['season_2011-12'] + (1.4809767007827759 <= (0.142857 + data[
        'season_2004-05']) / 2.0).astype(float)) / 2.0) + data['period_6'])
        ) / 2.0)) + np.tanh((-(data['action_type_Reverse Layup Shot'] > 2.0 -
        np.minimum(0.31831, (data['shot_type_3PT Field Goal'] > data[
        'season_1999-00'] * (data['minutes_remaining_0'] + (data[
        'action_type_Driving Finger Roll Shot'] + data['season_2004-05'])))
        .astype(float))).astype(float) + data['distance_0'] <= data[
        'minutes_remaining_10'] + (data['minutes_remaining_0'] + -2.83333) /
        2.0).astype(float)) + np.tanh(data['away_False'] * (data[
        'shot_zone_range_Back Court Shot'] - (data['opponent_SEA'] - np.
        minimum(data['season_2008-09'], (0.31831 * 0.06383 > (data[
        'season_1998-99'] - np.maximum(data['distance_45'], data[
        'action_type_Tip Shot'] * 2.68182)) * np.minimum(10.670692443847656,
        0.076923)).astype(float))))) + np.tanh(data[
        'action_type_Jump Hook Shot'] + data['shot_zone_area_Center(C)'] *
        (np.tanh(data['opponent_NOH']) + (np.maximum(data['last_moments_0'],
        data['action_type_Driving Dunk Shot']) + np.minimum(data[
        'opponent_HOU'] * -(data['period_2'] * 0.367879), 1.0)) / 2.0)
        ) + np.tanh(np.maximum(data['distance_41'], (data[
        'action_type_Driving Dunk Shot'] + (np.maximum(np.maximum(data[
        'season_1998-99'], np.maximum(data['action_type_Driving Dunk Shot'],
        1.570796 * data['action_type_Turnaround Bank shot'] + (data[
        'action_type_Driving Dunk Shot'] - data[
        'action_type_Driving Dunk Shot'] * np.tanh(0.586957)))), data[
        'action_type_Driving Dunk Shot']) + data['distance_15']) / 2.0) / 2.0)
        ) + np.tanh((data['action_type_Slam Dunk Shot'] + -((data[
        'action_type_Dunk Shot'] > data['last_moments_1'] * np.tanh(data[
        'action_type_Dunk Shot'])).astype(float) - data['period_2'] * np.
        tanh(0.31831 - np.minimum(data['season_2007-08'], 1.6))) + (data[
        'action_type_Reverse Dunk Shot'] - np.tanh(np.tanh(data[
        'opponent_TOR'])))) / 2.0) + np.tanh(((0.06383 * data['distance_43'
        ] + data['action_type_Turnaround Jump Shot']) / 2.0 + 1.21875) / 
        2.0 * np.minimum((data['shot_zone_basic_Mid-Range'] + (data[
        'action_type_Hook Shot'] + np.tanh(0.142857)) / 2.0) / 2.0 * data[
        'opponent_HOU'], np.maximum(0.31831, np.maximum(data['distance_35'],
        3.141593)) + np.tanh(1.0))) + np.tanh(np.tanh(np.maximum(0.540541,
        data['action_type_Alley Oop Dunk Shot'])) * ((0.63662 - (data[
        'shot_zone_area_Back Court(BC)'] + (data[
        'action_type_Finger Roll Shot'] + np.tanh(data['opponent_MIA'])) / 
        2.0) / 2.0 + data['opponent_VAN']) / 2.0)) + np.tanh(np.tanh(-(data
        ['action_type_Jump Shot'] * ((data['opponent_PHI'] + -data[
        'opponent_TOR']) / 2.0)) + np.minimum(data[
        'action_type_Alley Oop Dunk Shot'], data[
        'action_type_Jump Bank Shot'] + 0.235294))) + np.tanh(data[
        'action_type_Turnaround Bank shot'] + np.maximum((0.06383 <= np.
        maximum(np.maximum(np.maximum(data['action_type_Pullup Jump shot'],
        -((data['action_type_Driving Dunk Shot'] - -data[
        'action_type_Alley Oop Dunk Shot']) * np.tanh(data[
        'action_type_Running Bank shot']))), np.minimum(0.365854, data[
        'season_2013-14'])), data['action_type_Slam Dunk Shot'])).astype(
        float), np.maximum(data['action_type_Driving Dunk Shot'], np.tanh((
        data['action_type_Pullup Jump shot'] + data[
        'action_type_Driving Dunk Shot']) / 2.0)))) + np.tanh(data[
        'shot_zone_basic_In The Paint (Non-RA)'] * (data['playoffs_0'] <= 
        data['shot_zone_basic_Backcourt'] + np.maximum(data['opponent_DEN'],
        data['action_type_Turnaround Bank shot'] * np.maximum(np.minimum(
        data['action_type_Reverse Dunk Shot'], data[
        'action_type_Reverse Dunk Shot']), np.tanh(np.tanh(data[
        'opponent_ATL'] * 0.586957))) - 0.457627)).astype(float)) + np.tanh(
        data['lon'] * ((data['distance_11'] + (np.maximum(data[
        'opponent_MEM'], np.minimum(data['distance_13'], (2.718282 > (data[
        'shot_zone_basic_Restricted Area'] > (np.maximum(data[
        'opponent_MEM'], data['action_type_Driving Jump shot']) + float(6.0 >
        1.570796 + 1.570796)) / 2.0).astype(float)).astype(float))) + data[
        'opponent_IND'])) / 2.0)) + np.tanh(data['lat'] * np.maximum(data[
        'distance_19'], (data['action_type_Jump Bank Shot'] + data[
        'action_type_Bank Shot']) / 2.0 * data['last_moments_0'])) + np.tanh(
        (data['action_type_Driving Dunk Shot'] + (data[
        'minutes_remaining_1'] + data['shot_zone_area_Back Court(BC)'] * (
        2.718282 + np.maximum((0.711864 + (1.34211 <= (data[
        'action_type_Driving Slam Dunk Shot'] > 0.90625).astype(float)).
        astype(float)) / 2.0, (0.076923 > np.tanh(data['distance_27'])).
        astype(float)))) * data['opponent_UTA']) / 2.0) + np.tanh(data[
        'shot_zone_basic_In The Paint (Non-RA)'] * (2.83333 * ((data[
        'action_type_Fadeaway Bank shot'] + np.minimum(0.540541, (data[
        'opponent_PHX'] + data['distance_2'] * (data['distance_29'] <= np.
        tanh(-np.maximum(data['distance_45'], data['distance_32'])) - data[
        'distance_15']).astype(float)) / 2.0)) / 2.0) - data[
        'season_2000-01'] * np.minimum(data['minutes_remaining_3'], data[
        'minutes_remaining_3']))) + np.tanh(data[
        'action_type_Driving Dunk Shot'] - -(data['season_1998-99'] * (data
        ['shot_zone_area_Left Side(L)'] + data['distance_33'] * -(data[
        'shot_zone_basic_Backcourt'] - (3.0 + (np.minimum(-3.0, data[
        'distance_38']) + (data['action_type_Step Back Jump shot'] + (
        1.18667 <= np.tanh(9.413593292236328)).astype(float)) / 2.0) / 2.0 *
        data['distance_22']))))) + np.tanh(np.minimum((0.235294 + data[
        'season_2005-06'] * data['distance_28']) / 2.0, np.minimum((data[
        'lat'] + 1.42553) / 2.0, 0.367879 + -((data['action_type_Tip Shot'] +
        data['distance_36']) / 2.0)))) + np.tanh((-(data['season_2010-11'] *
        --data['shot_zone_area_Right Side Center(RC)']) + (data[
        'shot_zone_basic_In The Paint (Non-RA)'] > (data[
        'action_type_Driving Slam Dunk Shot'] + -data['period_5']) * data[
        'shot_zone_basic_In The Paint (Non-RA)']).astype(float)) / 2.0
        ) + np.tanh(np.minimum(2.83333, data['away_True'] * ((data[
        'action_type_Dunk Shot'] + (data['shot_type_3PT Field Goal'] <= 
        data['action_type_Jump Shot'] - (1.34211 + -((((data[
        'action_type_Reverse Layup Shot'] * 0.540541 <= data['distance_12']
        ).astype(float) > np.maximum(-0.076923, -5.0)).astype(float) + data
        ['opponent_BKN']) / 2.0)) / 2.0 * data['distance_16']).astype(float
        )) / 2.0))) + np.tanh(data['shot_zone_area_Center(C)'] * np.minimum
        (3.0, np.maximum(-((data['action_type_Slam Dunk Shot'] > np.tanh((
        data['action_type_Driving Slam Dunk Shot'] <= data[
        'action_type_Driving Slam Dunk Shot']).astype(float) * data[
        'distance_31'])).astype(float) * np.tanh(np.tanh(data[
        'action_type_Dunk Shot']))), (0.0 + np.maximum(np.maximum(data[
        'action_type_Driving Slam Dunk Shot'], np.tanh(data['distance_31'])
        ), np.maximum(data['action_type_Slam Dunk Shot'], data[
        'opponent_MIL']))) / 2.0))) + np.tanh(data['season_2011-12'] * ((
        0.210526 + (data['opponent_IND'] + (data['lon'] - -(data[
        'opponent_TOR'] * data['opponent_DEN']))) / 2.0) / 2.0) - data[
        'action_type_Tip Shot']) + np.tanh((0.142857 + (data[
        'shot_zone_area_Back Court(BC)'] + -(data['period_6'] + np.minimum(
        data['action_type_Finger Roll Layup Shot'], np.minimum(data[
        'season_2012-13'], --data['season_1999-00']))) - data['distance_25'
        ])) / 2.0 * (data['season_2011-12'] + np.tanh(np.tanh(data[
        'shot_zone_basic_Mid-Range']) + np.tanh(0.90625)))) + np.tanh(np.
        maximum(np.minimum((data['shot_zone_area_Right Side Center(RC)'] > 
        0.586957).astype(float), data['minutes_remaining_4']), data[
        'action_type_Slam Dunk Shot'] + (data['last_moments_0'] + (0.365854 -
        (0.586957 + np.maximum(1.0, (data['action_type_Running Layup Shot'] <=
        np.maximum(-(data['last_moments_1'] > 0.0).astype(float), data[
        'opponent_PHX'])).astype(float))) / 2.0)) / 2.0)) + np.tanh(data[
        'season_2000-01'] * ((data['shot_zone_area_Right Side Center(RC)'] +
        (data['minutes_remaining_8'] > data['distance_44'] * (data[
        'action_type_Tip Shot'] * (data['action_type_Tip Shot'] + np.
        minimum(np.tanh(np.minimum(data['action_type_Slam Dunk Shot'], np.
        maximum(float(0.142857 > 0.142857), 3.0)) - float(-0.90625 > 
        0.142857)), 1.42553)))).astype(float)) / 2.0)) + np.tanh(-(data[
        'action_type_Reverse Layup Shot'] - -np.tanh(np.maximum(np.maximum(
        data['distance_31'], data['action_type_Finger Roll Shot']), data[
        'shot_zone_area_Left Side Center(LC)'] * data[
        'action_type_Hook Shot'] * -np.minimum(0.63662, 0.711864 + 0.571429))))
        ) + np.tanh(data['distance_8'] * -(0.63662 - np.maximum(data[
        'action_type_Fadeaway Jump Shot'], np.maximum(data['season_1996-97'
        ], data['distance_9'] - (0.457627 > data['distance_9']).astype(
        float))))) + np.tanh(data['opponent_NYK'] * (data[
        'shot_zone_range_8-16 ft.'] - np.tanh(data['opponent_MIA'] * (data[
        'season_2008-09'] - data['distance_34'] * np.maximum(np.minimum(
        data['minutes_remaining_1'], data['action_type_Dunk']), data[
        'distance_37']))))) + np.tanh(data['season_2008-09'] * ((data[
        'shot_zone_area_Left Side Center(LC)'] + 1.0) / 2.0 + np.minimum(
        data['distance_5'], data['period_5'] - np.maximum(data[
        'shot_zone_area_Right Side(R)'], (2.0 <= -np.tanh(
        1.3872638940811157)).astype(float))))) + np.tanh(0.142857 * data[
        'action_type_Driving Dunk Shot'] - ((data['distance_10'] > np.tanh(
        data['distance_41']) * data['distance_35'] * np.tanh(np.maximum(
        data['action_type_Driving Dunk Shot'], np.tanh(2.83333)))).astype(
        float) + data['shot_zone_range_16-24 ft.'] * -data['opponent_ORL']) /
        2.0) + np.tanh(data['action_type_Driving Reverse Layup Shot'] + np.
        maximum(data['action_type_Running Bank shot'], np.maximum(data[
        'season_2009-10'], (data['action_type_Running Hook Shot'] + (data[
        'action_type_Driving Slam Dunk Shot'] + (1.173173427581787 <= (data
        ['season_2009-10'] * data['action_type_Running Bank shot'] <= 
        2.718282 * np.minimum(0.571429, (data[
        'action_type_Running Hook Shot'] > data['distance_29'] - data[
        'action_type_Driving Dunk Shot']).astype(float))).astype(float)).
        astype(float)) / 2.0) / 2.0))) + np.tanh((data['distance_6'] - np.
        minimum(2.83333, --data['period_3'])) * data['season_2000-01']
        ) + np.tanh(data['action_type_Running Jump Shot'] * (data[
        'action_type_Reverse Dunk Shot'] * (data[
        'action_type_Alley Oop Dunk Shot'] - np.maximum(data[
        'action_type_Alley Oop Dunk Shot'], data[
        'action_type_Reverse Dunk Shot'])) - (data[
        'shot_zone_basic_In The Paint (Non-RA)'] + np.tanh((data[
        'last_moments_1'] + data['action_type_Driving Finger Roll Shot']) /
        2.0)))) + np.tanh(-1.34211 * -(data[
        'action_type_Fadeaway Jump Shot'] * -((data['season_1996-97'] + (-(
        -data['action_type_Driving Finger Roll Layup Shot'] * (3.141593 - (
        data['away_True'] + (data['minutes_remaining_2'] + data[
        'distance_9'])))) + (data['shot_zone_range_16-24 ft.'] + np.minimum
        (3.0, np.minimum(0.367879, 6.148378849029541))))) / 2.0))) + np.tanh(
        data['period_1'] * np.maximum(-1.0, np.minimum(data['distance_17'] *
        np.tanh(np.minimum(data['distance_37'], data[
        'action_type_Driving Slam Dunk Shot']) - -((data[
        'action_type_Driving Slam Dunk Shot'] + data['distance_17']) / 2.0)
        ), (1.18667 > data['season_2014-15'] * -2.0).astype(float) - data[
        'minutes_remaining_5']))) + np.tanh(data[
        'action_type_Running Jump Shot'] * (data['distance_8'] <= np.
        minimum(data['action_type_Jump Hook Shot'], min(0.365854, float(
        0.540541 > 0.235294)))).astype(float) * np.tanh((math.tanh(math.
        tanh(1.21875)) > (data['season_1996-97'] + (1.570796 + data[
        'action_type_Pullup Jump shot']) / 2.0) / 2.0).astype(float) - data
        ['season_2008-09'])) + np.tanh(np.tanh(data['away_True'] * -np.tanh
        ((data['shot_zone_range_16-24 ft.'] > -(data[
        'action_type_Driving Reverse Layup Shot'] + (data[
        'shot_zone_area_Right Side(R)'] - np.maximum(data['period_3'], data
        ['playoffs_0'])))).astype(float) * np.tanh((-data[
        'shot_zone_area_Right Side Center(RC)'] > np.tanh(0.367879)).astype
        (float))))) + np.tanh(data['action_type_Driving Slam Dunk Shot'] + 
        data['opponent_TOR'] * -(data['action_type_Driving Dunk Shot'] - np
        .tanh(data['shot_zone_area_Left Side Center(LC)']))) + np.tanh((
        data['action_type_Turnaround Bank shot'] + -(data[
        'action_type_Turnaround Jump Shot'] * data[
        'shot_zone_basic_Mid-Range'])) / 2.0) + np.tanh((data[
        'opponent_NYK'] > (data['distance_27'] + (data['season_2004-05'] > 
        (0.210526 + np.tanh(data['action_type_Driving Slam Dunk Shot'] *
        data['action_type_Driving Finger Roll Layup Shot'])) / 2.0).astype(
        float)) * (data['distance_38'] + np.minimum(data[
        'shot_zone_basic_Mid-Range'], 0.90625))).astype(float)) + np.tanh(-
        data['opponent_CHI'] * ((np.minimum(0.142857, -np.tanh(data[
        'distance_8'])) + ((data['shot_zone_area_Left Side(L)'] + (data[
        'action_type_Reverse Dunk Shot'] - (data[
        'action_type_Reverse Dunk Shot'] + np.tanh(-0.365854)) / 2.0)) / 
        2.0 + data['action_type_Driving Layup Shot'])) / 2.0) * -2.0
        ) + np.tanh(-((data['playoffs_1'] + np.maximum(-1.21875 - (-data[
        'action_type_Bank Shot'] + -(0.367879 - 3.0)) / 2.0, np.tanh(data[
        'season_2004-05']))) / 2.0 * data['minutes_remaining_6'])) + np.tanh(
        data['action_type_Jump Shot'] * (data['opponent_NJN'] + (data[
        'distance_16'] > np.tanh(1.0)).astype(float))) + np.tanh(data[
        'season_2002-03'] * -((np.minimum(1.21875, data[
        'shot_zone_area_Center(C)']) + np.tanh(data[
        'action_type_Fadeaway Bank shot'])) / 2.0)) + np.tanh(data[
        'shot_zone_range_Less Than 8 ft.'] * ((0.367879 * (data[
        'opponent_MIL'] * (2.83333 <= data['opponent_MIL'] + data[
        'action_type_Fadeaway Jump Shot']).astype(float)) + np.maximum(-2.0,
        np.tanh(data['period_4']))) / 2.0)) + np.tanh((data['distance_3'] -
        data['action_type_Driving Layup Shot']) * data['season_1999-00']
        ) + np.tanh(data['shot_zone_range_8-16 ft.'] * (data[
        'action_type_Pullup Jump shot'] > data[
        'action_type_Alley Oop Dunk Shot'] - (data[
        'action_type_Fadeaway Bank shot'] + (-(data[
        'action_type_Alley Oop Dunk Shot'] > data['season_2000-01']).astype
        (float) > data['opponent_NOH']).astype(float)) / 2.0).astype(float)
        ) + np.tanh(np.maximum(data['action_type_Slam Dunk Shot'] - (np.
        minimum(0.457627, np.tanh(2.718282)) + (float(1.6 <= 3.0) - np.
        maximum(data['distance_14'], data['action_type_Reverse Dunk Shot'])
        )) / 2.0, (data['distance_1'] > (-data[
        'shot_zone_area_Right Side(R)'] > data[
        'action_type_Alley Oop Dunk Shot']).astype(float)).astype(float))
        ) + np.tanh(-(data['distance_24'] * np.minimum(np.tanh(data[
        'minutes_remaining_0']) - (0.076923 + (data[
        'action_type_Finger Roll Layup Shot'] > data[
        'action_type_Jump Bank Shot']).astype(float)) / 2.0, (0.365854 <= (
        0.0 + data['period_3'] * -2.0) / 2.0).astype(float)) + data[
        'action_type_Driving Reverse Layup Shot'] * (1.570796 <= np.maximum
        (3.0, 0.06383)).astype(float))) + np.tanh(np.maximum(data[
        'action_type_Fadeaway Bank shot'] * 2.718282, data[
        'shot_zone_range_Less Than 8 ft.'] * data['minutes_remaining_10']))
    return Outputs(0.1 * predictions)


def main():
    df = pd.read_csv('../input/data.csv')
    df.drop(['game_event_id', 'game_id', 'team_id', 'team_name'], axis=1,
        inplace=True)
    df.sort_values('game_date', inplace=True)
    mask = df['shot_made_flag'].isnull()
    actiontypes = dict(df.action_type.value_counts())
    df['type'] = df.apply(lambda row: row['action_type'] if actiontypes[row
        ['action_type']] > 20 else row['combined_shot_type'], axis=1)
    df.drop(['action_type', 'combined_shot_type'], axis=1, inplace=True)
    df['away'] = df.matchup.str.contains('@')
    df.drop('matchup', axis=1, inplace=True)
    df['distance'] = df.apply(lambda row: row['shot_distance'] if row[
        'shot_distance'] < 45 else 45, axis=1)
    df['time_remaining'] = df.apply(lambda x: x['minutes_remaining'] * 60 +
        x['seconds_remaining'], axis=1)
    df['last_moments'] = df.apply(lambda row: 1 if row['time_remaining'] < 
        3 else 0, axis=1)
    data = pd.get_dummies(df['type'], prefix='action_type')
    features = ['away', 'period', 'playoffs', 'shot_type', 'shot_zone_area',
        'shot_zone_basic', 'season', 'shot_zone_range', 'opponent',
        'distance', 'minutes_remaining', 'last_moments']
    for f in features:
        data = pd.concat([data, pd.get_dummies(df[f], prefix=f)], axis=1)
    data['lat'] = df['lat']
    data['lon'] = df['lon']
    ss = StandardScaler()
    train = data[~mask].copy()
    features = train.columns
    train[features] = np.round(ss.fit_transform(train[features]), 6)
    train['shot_made_flag'] = df.shot_made_flag[~mask]
    test = data[mask].copy()
    test.insert(0, 'shot_id', df[mask].shot_id)
    test[features] = np.round(ss.transform(test[features]), 6)
    ss = StandardScaler()
    train = data[~mask].copy()
    features = train.columns
    train[features] = np.round(ss.fit_transform(train[features]), 6)
    train['shot_made_flag'] = df.shot_made_flag[~mask]
    test = data[mask].copy()
    test.insert(0, 'shot_id', df[mask].shot_id)
    test[features] = np.round(ss.transform(test[features]), 6)
    trainpredictions1 = GPIndividual1(train)
    trainpredictions2 = GPIndividual2(train)
    testpredictions1 = GPIndividual1(test)
    testpredictions2 = GPIndividual2(test)
    predictions = (trainpredictions1 + trainpredictions2) / 2.0
    print(log_loss(train.shot_made_flag.values, predictions.values))
    predictions = (testpredictions1 + testpredictions2) / 2
    submission = pd.DataFrame({'shot_id': test.shot_id, 'shot_made_flag':
        predictions})
    submission.sort_values('shot_id', inplace=True)
    submission.to_csv('arisubmission.csv', index=False)
    predictions = np.power(trainpredictions1 * trainpredictions2, 1.0 / 2)
    print(log_loss(train.shot_made_flag.values, predictions.values))
    predictions = np.power(testpredictions1 * testpredictions2, 1.0 / 2)
    submission = pd.DataFrame({'shot_id': test.shot_id, 'shot_made_flag':
        predictions})
    submission.sort_values('shot_id', inplace=True)
    submission.to_csv('geosubmission.csv', index=False)


if __name__ == '__main__':
    main()
if '__name__' not in TANGSHAN:
    import csv
    if isinstance(__name__, np.ndarray) or isinstance(__name__, pd.DataFrame
        ) or isinstance(__name__, pd.Series):
        shape_size = __name__.shape
    elif isinstance(__name__, list):
        shape_size = len(__name__)
    else:
        shape_size = 'any'
    check_type = type(__name__)
    with open('tas.csv', 'a+') as f:
        TANGSHAN.append('__name__')
        writer = csv.writer(f)
        writer.writerow(['__name__', 410, check_type, shape_size])