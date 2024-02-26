import matplotlib.pyplot as plt
import random
import fastf1.plotting
import numpy

# enable some matplotlib patches for plotting timedelta values and load
# FastF1's default color scheme
# fastf1.Cache.clear_cache()
fastf1.plotting.setup_mpl(misc_mpl_mods=False)

# load a session and its telemetry data
session = fastf1.get_session(2023, 'Las Vegas Grand Prix', 'qualifying')
session.load()

fig, ax = plt.subplots()

laps_ver = session.laps.pick_driver('VER')

fastest_lap = laps_ver.pick_fastest().get_car_data().add_distance()
lap = fastest_lap


ax.plot(lap['Time'].dt.total_seconds(), lap['Speed'], color='white', label='VER')


ax.set_xlabel('time')
ax.set_ylabel('speed')


plt.suptitle(f"Speed \n "
                f"{session.event['EventName']} {session.event.year}")

plt.show()