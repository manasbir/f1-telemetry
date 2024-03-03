import csv
import matplotlib.pyplot as plt
import fastf1.plotting
import numpy as np
import pandas as pd
import os


# enable some matplotlib patches for plotting timedelta values and load
# FastF1's default color scheme
# fastf1.Cache.clear_cache()
fastf1.plotting.setup_mpl(misc_mpl_mods=False)
max_engine_power = 661949
fig, ax = plt.subplots()
f1_car_weight = 830
normal_force = f1_car_weight * 9.81
rpm_max = 15_000
avg_air_density_div_2 = 0.6

teams = {
    'Alfa Romeo': ['BOT', 'ZHO'],
    'Alpha Tauri': ['RIC', 'DEV', 'TSU', 'LAW'],
    'Alpine': ['OCO', 'GAS'],
    'Aston Martin': ['ALO', 'STR'],
    'Ferrari': ['LEC', 'SAI'],
    'Haas': ['MAG', 'HUL'],
    'McLaren': ['PIA', 'NOR'],
    'Mercedes': ['HAM', 'RUS'],
    'Red Bull': ['VER', 'PER'],
    'williams': ['ALB', 'SAR']}

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

data = {}
drs_data = {}

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
                if row['Acceleration'] < 0 or row['RPM'] < 11000 or row['Throttle'] != 100 or row['nGear'] != 8:
                    lap.drop(index, inplace=True)
                else:
                    lap.loc[index, 'v^2 (m^2/s^4)'] = (row['Speed'] ** 2)
                    lap.loc[index, 'F_lost (N)'] = (((row['RPM']/rpm_max) * max_engine_power)/row['Speed']) - (f1_car_weight * row['Acceleration'])

            if team not in data:
                data[team][race] = lap
                drs_data[team][race] = drs_lap
                if 'all' not in data[team].values():
                    data[team]['all'] = lap
                    drs_data[team]['all'] = drs_lap
            else:
                data[team][race] = pd.concat([data[team][race], lap], ignore_index=True)
                drs_data[team][race] = pd.concat([drs_data[team][race], lap], ignore_index=True)
                data[team]['all'] = pd.concat([data[team][race], lap], ignore_index=True)
                drs_data[team]['all'] = pd.concat([drs_data[team][race], lap], ignore_index=True)

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


regressions_data = []
drs_regressions_data = []

for team, lap_data in data.items():
    plt.scatter(lap_data['v^2 (m^2/s^4)'], lap_data['F_lost (N)'], label=team)

    # Perform linear regression
    coeffs = np.polyfit(lap_data['v^2 (m^2/s^4)'], lap_data['F_lost (N)'], 1)
    regression_line = np.poly1d(coeffs)

    # Plot regression line
    plt.plot(lap_data['v^2 (m^2/s^4)'], regression_line(lap_data['v^2 (m^2/s^4)']), label=f'Regression Line ({team})')

    # Display equation of regression line
    plt.text(0.1, 0.9, f"y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}", transform=ax.transAxes)
    
    # Calculate coefficients
    c_d_mul_a = coeffs[0] / avg_air_density_div_2
    c_rr = coeffs[1] / (normal_force * 4)

    # Append data to list
    regressions_data.append({'Team': team, 'Regression_Line': regression_line, 'C_d_mul_A': c_d_mul_a, 'C_rr': c_rr})

# plt.xlabel('v^2 (m^2/s^4)')
# plt.ylabel('F_lost (N)')
# plt.title('F_lost (N) vs v^2 (m^2/s^4) for Each Team with Regression Lines')
# plt.legend()
# plt.show()

for team, drs_lap_data in drs_data.items():
    plt.scatter(drs_lap_data['v^2 (m^2/s^4)'], drs_lap_data['F_lost (N)'], label=team)

    # Perform linear regression
    coeffs = np.polyfit(drs_lap_data['v^2 (m^2/s^4)'], drs_lap_data['F_lost (N)'], 1)
    regression_line = np.poly1d(coeffs)

    # Plot regression line
    plt.plot(drs_lap_data['v^2 (m^2/s^4)'], regression_line(drs_lap_data['v^2 (m^2/s^4)']), label=f'Regression Line ({team})')

    # Display equation of regression line
    plt.text(0.1, 0.9, f"y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}", transform=ax.transAxes)
    
    # Calculate coefficients
    c_d_mul_a = coeffs[0] / avg_air_density_div_2
    c_rr = coeffs[1] / (normal_force * 4)

    # Append data to list
    drs_regressions_data.append({'Team': team, 'Regression_Line': regression_line, 'C_d_mul_A': c_d_mul_a, 'C_rr': c_rr})

# plt.xlabel('v^2 (m^2/s^4)')
# plt.ylabel('F_lost (N)')
# plt.title('F_lost (N) vs v^2 (m^2/s^4) for Each Team with Regression Lines (DRS)')
# plt.legend()
# plt.show()

# Create DataFrames from lists of dictionaries
regressions_df = pd.DataFrame(regressions_data)
drs_regressions_df = pd.DataFrame(drs_regressions_data)

regressions_df.to_csv('regressions_data.csv', index=False)
drs_regressions_df.to_csv('drs_regressions_data.csv', index=False)
