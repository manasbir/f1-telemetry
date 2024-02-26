import csv
import matplotlib.pyplot as plt
import fastf1.plotting
import numpy as np

# enable some matplotlib patches for plotting timedelta values and load
# FastF1's default color scheme
# fastf1.Cache.clear_cache()
fastf1.plotting.setup_mpl(misc_mpl_mods=False)

# load a session and its telemetry data
session = fastf1.get_session(2023, 'Las Vegas Grand Prix', 'qualifying')
session.load()

fig, ax = plt.subplots()

laps_ver = session.laps.pick_driver('VER')
laps_per = session.laps.pick_driver('PER')

lap_ver = laps_ver.pick_fastest().get_car_data().add_distance()
lap_per = laps_per.pick_fastest().get_car_data().add_distance()

# Calculate speed for VER
time_ver = lap_ver['Time'].dt.total_seconds()
speed_ver = lap_ver['Speed'] / 3.6  # Convert speed from km/h to m/s

# Compute derivative of speed (rate of change of speed with respect to time) for VER
acceleration_ver = np.gradient(speed_ver, time_ver)

# Calculate speed for PER
time_per = lap_per['Time'].dt.total_seconds()
speed_per = lap_per['Speed'] / 3.6  # Convert speed from km/h to m/s

# Compute derivative of speed (rate of change of speed with respect to time) for PER
acceleration_per = np.gradient(speed_per, time_per)

# Get time, velocity, and rpm data for VER
velocity_ver = speed_ver
rpm_ver = lap_ver['RPM']
distance_ver = lap_ver['Distance']

# Get time, velocity, and rpm data for PER
velocity_per = speed_per
rpm_per = lap_per['RPM']
distance_per = lap_per['Distance']

# Export data to CSV for positive acceleration values with RPM > 10000 and distance not between 750 & 1500
positive_acceleration_data = []
for i, acc_ver in enumerate(acceleration_ver):
    if acc_ver > 0 and rpm_ver[i] > 10000 and not (750 < distance_ver[i] < 1500) and not (4200 < distance_ver[i] < 5100):
        positive_acceleration_data.append([time_ver[i], velocity_ver[i], acc_ver, rpm_ver[i]])

for i, acc_per in enumerate(acceleration_per):
    if acc_per > 0 and rpm_per[i] > 10000 and not (750 < distance_per[i] < 1500) and not (4200 < distance_per[i] < 5100):
        positive_acceleration_data.append([time_per[i], velocity_per[i], acc_per, rpm_per[i]])

# Write positive acceleration data to CSV file
with open('positive_acceleration_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Time', 'Velocity', 'Acceleration', 'RPM'])
    writer.writerows(positive_acceleration_data)

# Plot acceleration for VER
ax.plot(time_ver, acceleration_ver, color='blue', label='VER Acceleration')

# Plot acceleration for PER
ax.plot(time_per, acceleration_per, color='red', label='PER Acceleration')

ax.set_xlabel('Time in s')
ax.set_ylabel('Acceleration')

plt.suptitle(f"Acceleration \n {session.event['EventName']} {session.event.year}")
plt.legend()
plt.show()
