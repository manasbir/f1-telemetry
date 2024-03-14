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

laps_ver = session.laps.pick_driver('TSU')

fastest_lap = laps_ver.pick_fastest().get_car_data().add_distance()
lap = fastest_lap

acceleration = numpy.gradient(lap['Speed']/3.6, lap['Time'].dt.total_seconds())


ax.plot(lap['Time'], lap["DRS"], color='white', label='TSU')


ax.set_xlabel('time')
ax.set_ylabel('speed')


plt.suptitle(f"acc \n "
                f"{session.event['EventName']} {session.event.year}")

plt.show()