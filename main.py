import csv
import matplotlib.pyplot as plt
import fastf1.plotting
import numpy as np
import pandas as pd
import os


# enable some matplotlib patches for plotting timedelta values and load
# FastF1's default color scheme
#fastf1.Cache.clear_cache()
fastf1.plotting.setup_mpl(misc_mpl_mods=False)
max_engine_power = 661949
fig, ax = plt.subplots()
f1_car_weight = 830
normal_force = f1_car_weight * 9.81
rpm_max = 15_000
avg_air_density_div_2 = 0.6

teams = {
    'Alfa Romeo': ['BOT', 'ZHO'],
    'AlphaTauri': ['RIC', 'DEV', 'TSU', 'LAW'],
    'Alpine': ['OCO', 'GAS'],
    'Aston Martin': ['ALO', 'STR'],
    'Ferrari': ['LEC', 'SAI'],
    'Haas': ['MAG', 'HUL'],
    'McLaren': ['PIA', 'NOR'],
    'Mercedes': ['HAM', 'RUS'],
    'Red Bull': ['VER', 'PER'],
    'Williams': ['ALB', 'SAR']}

races = [
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",
    "Australian Grand Prix",
    "Azerbaijan Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Spanish Grand Prix",
    "Canadian Grand Prix",
    "Austrian Grand Prix",
    "British Grand Prix",
    "Hungarian Grand Prix",
    "Belgian Grand Prix",
    "Dutch Grand Prix",
    "Italian Grand Prix",
    "Singapore Grand Prix",
    "Japanese Grand Prix",
    "Qatar Grand Prix",
    "United States Grand Prix",
    "Mexican Grand Prix",
    "Brazilian Grand Prix",
    "Las Vegas Grand Prix",
    "Abu Dhabi Grand Prix"
]

data = {
    'Alfa Romeo': {},
    'AlphaTauri': {},
    'Alpine': {},
    'Aston Martin': {},
    'Ferrari': {},
    'Haas': {},
    'McLaren': {},
    'Mercedes': {},
    'Red Bull': {},
    'Williams': {}
}
drs_data = {
    'Alfa Romeo': {},
    'AlphaTauri': {},
    'Alpine': {},
    'Aston Martin': {},
    'Ferrari': {},
    'Haas': {},
    'McLaren': {},
    'Mercedes': {},
    'Red Bull': {},
    'Williams': {}
}

team_to_car = {
    'Red Bull': 'RB19',
    'Ferrari': 'SF-23',
    'Mercedes': 'W14',
    'Alpine': 'A523',
    'McLaren': 'MCL60',
    'Alfa Romeo': 'C43',
    'Aston Martin': 'AMR23',
    'Haas': 'VF-23',
    'AlphaTauri': 'AT04',
    'Williams': 'FW45'
}


for race in races:
    session = fastf1.get_session(2023, race, 'qualifying')
    session.load()
    for team in teams:

        for driver in teams[team]:
            laps = session.laps.pick_driver(driver)
            try:
                lap = laps.pick_fastest().get_car_data().add_distance()
            except:
                continue
            
            if driver == ['VER'] and race == 'Saudi Arabian Grand Prix':
                continue

            start_time = lap['Date'][0]
            lap['Speed'] /= 3.6
            lap['Time'] = (lap['Date'] - start_time).dt.total_seconds()
            lap['Acceleration'] = np.gradient(lap['Speed'], lap['Time'])

            drs_lap = lap.copy()

            for index, row in drs_lap.iterrows():
                if row['DRS'] == 12 or row['DRS'] == 14:
                    drs_lap.loc[index, 'v^2 (m^2/s^4)'] = row['Speed'] ** 2
                    drs_lap.loc[index, 'F_lost (N)'] = (((row['RPM']/rpm_max) * max_engine_power)/row['Speed']) - (f1_car_weight * row['Acceleration'])
                    lap.drop(index, inplace=True)
                else:
                    drs_lap.drop(index, inplace=True)

            
            for index, row in lap.iterrows():
                if row['Acceleration'] < 0 or row['RPM'] < 11000 or row['Throttle'] != 100:
                    lap.drop(index, inplace=True)
                else:
                    lap.loc[index, 'v^2 (m^2/s^4)'] = (row['Speed'] ** 2)
                    lap.loc[index, 'F_lost (N)'] = (((row['RPM']/rpm_max) * max_engine_power)/row['Speed']) - (f1_car_weight * row['Acceleration'])

            if 'all' not in data[team]:
                data[team]['all'] = lap
                drs_data[team]['all'] = drs_lap
            else:
                data[team]['all'] = pd.concat([data[team]['all'], lap], ignore_index=True)
                drs_data[team]['all'] = pd.concat([drs_data[team]['all'], drs_lap], ignore_index=True)
            
            if race not in data[team]:
                data[team][race] = lap
                drs_data[team][race] = drs_lap
            else:
                data[team][race] = pd.concat([data[team][race], lap], ignore_index=True)
                drs_data[team][race] = pd.concat([drs_data[team][race], drs_lap], ignore_index=True)

            

output_directory = "team_data"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Export non-DRS data
for team in data:
    output_folder = os.path.join(output_directory, team)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data[team]['all'].to_csv(os.path.join(output_folder, "non_drs_data.csv"), index=False)

# Export DRS data
for team in drs_data:
    output_folder = os.path.join(output_directory, team)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    drs_data[team]['all'].to_csv(os.path.join(output_folder, "drs_data.csv"), index=False)


all_data = {
    'Alfa Romeo': {},
    'AlphaTauri': {},
    'Alpine': {},
    'Aston Martin': {},
    'Ferrari': {},
    'Haas': {},
    'McLaren': {},
    'Mercedes': {},
    'Red Bull': {},
    'Williams': {}
}


for team in data:
    #plt.scatter(data[team]['all']['v^2 (m^2/s^4)'], data[team]['all']['F_lost (N)'], label=team)
    regressions_data = []
    for race in data[team]:
        # Perform linear regression
        try:
            coeffs = np.polyfit(data[team][race]['v^2 (m^2/s^4)'], data[team][race]['F_lost (N)'], 1)
        except:
            print("skipped " + race + ' for '+ team + ' no DRS')
            continue

        regression_line = np.poly1d(coeffs)

        # Plot regression line
        #plt.plot(data[team][race]['v^2 (m^2/s^4)'], regression_line(data[team][race]['v^2 (m^2/s^4)']), label=f'Regression Line ({team})')
        # Display equation of regression line
        #plt.text(0.1, 0.9, f"y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}", transform=ax.transAxes)
        
        # Calculate coefficients
        c_d_mul_a = coeffs[0] / avg_air_density_div_2
        c_rr = coeffs[1] / (normal_force * 4)

        if race == 'all':
            all_data[team]['c_d_mul_a'] = c_d_mul_a
            all_data[team]['c_rr'] = c_rr


        # Append data to list
        regressions_data.append({'Race': race, 'Regression_Line': regression_line, 'C_d_mul_A': c_d_mul_a, 'C_rr': c_rr})
        regressions_df = pd.DataFrame(regressions_data)
        regressions_df.to_csv('team_data/' + team +'/regressions_data.csv', index=False)

for team in drs_data:
    #plt.scatter(data[team]['all']['v^2 (m^2/s^4)'], data[team]['all']['F_lost (N)'], label=team)
    drs_regressions_data = []
    for race in drs_data[team]:
        # Perform linear regression
    
        try:
            drs_coeffs = np.polyfit(drs_data[team][race]['v^2 (m^2/s^4)'], drs_data[team][race]['F_lost (N)'], 1)
        except:
            print("skipped " + race + ' for '+ team + ' DRS')
            continue

        drs_regression_line = np.poly1d(drs_coeffs)

        # Plot regression line
        #plt.plot(data[team][race]['v^2 (m^2/s^4)'], regression_line(data[team][race]['v^2 (m^2/s^4)']), label=f'Regression Line ({team})')
        # Display equation of regression line
        #plt.text(0.1, 0.9, f"y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}", transform=ax.transAxes)
        
        # Calculate coefficients

        drs_c_d_mul_a = drs_coeffs[0] / avg_air_density_div_2
        drs_c_rr = drs_coeffs[1] / (normal_force * 4)

        if race == 'all':
            all_data[team]['drs_c_d_mul_a'] = drs_c_d_mul_a
            all_data[team]['drs_c_rr'] = drs_c_rr

        # Append data to list
        drs_regressions_data.append({'Race': race, 'Regression_Line': drs_regression_line, 'C_d_mul_A': drs_c_d_mul_a, 'C_rr': drs_c_rr})
        drs_regressions_df = pd.DataFrame(drs_regressions_data)
        drs_regressions_df.to_csv('team_data/' + team +'/drs_regressions_data.csv', index=False)


# plt.xlabel('v^2 (m^2/s^4)')
# plt.ylabel('F_lost (N)')
# plt.title('F_lost (N) vs v^2 (m^2/s^4) for Each Team with Regression Lines')
# plt.legend()
# plt.show()

for team in drs_data:
    plt.scatter(drs_data[team]['Italian Grand Prix']['v^2 (m^2/s^4)'], drs_data[team]['Italian Grand Prix']['F_lost (N)'], label=team_to_car[team])


plt.xlabel('v^2 (m^2/s^4)')
plt.ylabel('F_lost (N)')
plt.title('F_lost (N) vs v^2 (m^2/s^4) for Each Car with DRS in the Italian Grand Prix', fontsize=10)
plt.legend()
plt.savefig('ITALY-ALL-DRS.png', bbox_inches='tight')

plt.show()

for team in data:
    plt.scatter(data[team]['Italian Grand Prix']['v^2 (m^2/s^4)'], data[team]['Italian Grand Prix']['F_lost (N)'], label=team_to_car[team])


plt.xlabel('v^2 (m^2/s^4)')
plt.ylabel('F_lost (N)')
plt.title('F_lost (N) vs v^2 (m^2/s^4) for Each Car without DRS in the Italian Grand Prix', fontsize=10)
plt.legend()
plt.savefig('ITALY-ALL-NO-DRS.png', bbox_inches='tight')
plt.show()


"""
for team, lap_data in data.items():
    plt.scatter(lap_data['v^2 (m^2/s^4)'], lap_data['F_lost (N)'], label=team)

plt.xlabel('v^2 (m^2/s^4)')
plt.ylabel('F_lost (N)')
plt.title('F_lost (N) vs v^2 (m^2/s^4) for Each Team')
plt.legend()
plt.show()


for team, lap_data in drs_data.items():
    plt.scatter(lap_data['v^2 (m^2/s^4)'], lap_data['F_lost (N)'], label=team)

plt.xlabel('v^2 (m^2/s^4)')
plt.ylabel('F_lost (N)')
plt.title('F_lost (N) vs v^2 (m^2/s^4) for Each Team')
plt.legend()
plt.show() """


plt.scatter(data['Red Bull']['Italian Grand Prix']['v^2 (m^2/s^4)'], data['Red Bull']['Italian Grand Prix']['F_lost (N)'], label='RB19')
plt.xlabel('v^2 (m^2/s^4)')
plt.ylabel('F_lost (N)')
plt.title('F_lost (N) vs v^2 (m^2/s^4) for the RB19 without DRS in the Italian Grand Prix', fontsize=10)
plt.legend()
plt.savefig('ITALY-RB19-NO-DRS.png', bbox_inches='tight')
plt.show()

plt.scatter(drs_data['Red Bull']['Italian Grand Prix']['v^2 (m^2/s^4)'], drs_data['Red Bull']['Italian Grand Prix']['F_lost (N)'], label='RB19')
plt.xlabel('v^2 (m^2/s^4)')
plt.ylabel('F_lost (N)')
plt.title('F_lost (N) vs v^2 (m^2/s^4) for the RB19 with DRS in the Italian Grand Prix', fontsize=10)
plt.legend()
plt.savefig('ITALY-RB19-DRS.png', bbox_inches='tight')
plt.show()

all_drs_data = {
    'Alfa Romeo': {},
    'AlphaTauri': {},
    'Alpine': {},
    'Aston Martin': {},
    'Ferrari': {},
    'Haas': {},
    'McLaren': {},
    'Mercedes': {},
    'Red Bull': {},
    'Williams': {}
}

for team in all_data:
    all_data[team]['cd_mul_a_diff'] = all_data[team]['c_d_mul_a'] - all_data[team]['drs_c_d_mul_a']
    all_data[team]['c_rr_diff'] = all_data[team]['c_rr'] - all_data[team]['drs_c_rr']
    all_data[team]['car'] = team_to_car[team]
    all_data_df = pd.DataFrame(all_data[team])
    all_data_df.to_csv('all_data.csv', index=False)


