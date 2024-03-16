import csv
import matplotlib.pyplot as plt
import fastf1.plotting
import numpy as np
import pandas as pd
import os
import logging
from scipy.optimize import fsolve

# Disable INFO level logs for all loggers
logging.getLogger('fastf1').setLevel(logging.WARNING)


def power_fn(x):
    return [max_engine_power/(z) for z in x]

def linear_to_quadratic(regression, x):
    return np.poly1d([regression.c[0], 0, regression.c[1]])(x)

def fn_to_solve(x, regression):
    if isinstance(x, np.int64):

        return regression.c[0]*(x**2) + regression.c[1] - ((12.5/15)*max_engine_power/x)
    else:

        return [regression.c[0]*(z**2) + regression.c[1] - ((12.5/15)*max_engine_power/z) for z in x]


# enable some matplotlib patches for plotting timedelta values and load
# FastF1's default color scheme
# fastf1.Cache.clear_cache()
fastf1.plotting.setup_mpl(misc_mpl_mods=False)
max_engine_power = 805_000  # in watts
fig, ax = plt.subplots()
f1_car_weight = 805
normal_force = f1_car_weight * 9.81
rpm_max = 15_000

teams = {
    "Alfa Romeo": ["BOT", "ZHO"],
    "AlphaTauri": ["RIC", "DEV", "TSU", "LAW"],
    "Alpine": ["OCO", "GAS"],
    "Aston Martin": ["ALO", "STR"],
    "Ferrari": ["LEC", "SAI"],
    "Haas": ["MAG", "HUL"],
    "McLaren": ["PIA", "NOR"],
    "Mercedes": ["HAM", "RUS"],
    "Williams": ["ALB", "SAR"],
    "Red Bull": ["VER", "PER"],
}
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
    "Abu Dhabi Grand Prix",
]
data = {
    "Alfa Romeo": {},
    "AlphaTauri": {},
    "Alpine": {},
    "Aston Martin": {},
    "Ferrari": {},
    "Haas": {},
    "McLaren": {},
    "Mercedes": {},
    "Williams": {},
    "Red Bull": {},
}
drs_data = {
    "Alfa Romeo": {},
    "AlphaTauri": {},
    "Alpine": {},
    "Aston Martin": {},
    "Ferrari": {},
    "Haas": {},
    "McLaren": {},
    "Mercedes": {},
    "Williams": {},
    "Red Bull": {},
}
team_to_car = {
    "Red Bull": "RB19",
    "Ferrari": "SF-23",
    "Mercedes": "W14",
    "Alpine": "A523",
    "McLaren": "MCL60",
    "Alfa Romeo": "C43",
    "Aston Martin": "AMR23",
    "Haas": "VF-23",
    "AlphaTauri": "AT04",
    "Williams": "FW45",
}
team_to_rank = {
    "Red Bull": "1",
    "Ferrari": "3",
    "Mercedes": "2",
    "Alpine": "6",
    "McLaren": "4",
    "Alfa Romeo": "9",
    "Aston Martin": "5",
    "Haas": "10",
    "AlphaTauri": "8",
    "Williams": "7",
}


for race in races:
    session = fastf1.get_session(2023, race, "qualifying")
    session.load()
    for team in teams:
        for driver in teams[team]:
            laps = session.laps.pick_driver(driver)
            try:
                lap = laps.pick_fastest().get_car_data()
            except:
                continue

            # if driver == ["VER"] and race == "Saudi Arabian Grand Prix":
            #     continue

            start_time = lap["Date"][0]
            lap["Speed"] /= 3.6
            lap["Time"] = (lap["Date"] - start_time).dt.total_seconds()
            lap["Acceleration"] = np.gradient(lap["Speed"], lap["Time"])

            drs_lap = lap.copy()

            for index, row in drs_lap.iterrows():
                if row["DRS"] == 12:
                    if (
                        row["Acceleration"] > 0
                        and row["RPM"] >= 10500
                        and row["Throttle"] >= 97
                        and row["Brake"] == False
                        and row["nGear"] == 8
                    ):
                        F_dissipated = (
                        ((row["RPM"] / rpm_max) * max_engine_power) / row["Speed"]
                        ) - (f1_car_weight * row["Acceleration"])
                        if F_dissipated < 0:
                            drs_lap.drop(index, inplace=True)
                            continue

                        drs_lap.loc[index, "v^2 (m^2/s^4)"] = row["Speed"] ** 2
                        drs_lap.loc[index, "F_dissipated (N)"] = (
                            ((row["RPM"] / rpm_max) * max_engine_power) / row["Speed"]
                        ) - (f1_car_weight * row["Acceleration"])
                    else:
                        drs_lap.drop(index, inplace=True)
                    lap.drop(index, inplace=True)
                else:
                    drs_lap.drop(index, inplace=True)

            for index, row in lap.iterrows():
                if (
                    row["Acceleration"] <= 0
                    or row["RPM"] < 10500
                    or row["Throttle"] < 97
                    or row["Brake"] == True
                    or row["nGear"] != 8
                    or row["DRS"] == 12
                    or row["DRS"] == 14
                    or row["DRS"] == 10
                ):
                    lap.drop(index, inplace=True)
                else:
                    F_dissipated = (
                        ((row["RPM"] / rpm_max) * max_engine_power) / row["Speed"]
                    ) - (f1_car_weight * row["Acceleration"])
                    if F_dissipated < 0:
                        lap.drop(index, inplace=True)
                        continue
                    if row["Speed"] > 95:
                        print(driver + " " + race + " " + str(row["Speed"]*3.6))
                    lap.loc[index, "F_dissipated (N)"] = F_dissipated
                    
                    lap.loc[index, "v^2 (m^2/s^4)"] = row["Speed"] ** 2
                    


            if "all" not in data[team]:
                data[team]["all"] = lap
                drs_data[team]["all"] = drs_lap
            else:
                data[team]["all"] = pd.concat(
                    [data[team]["all"], lap], ignore_index=True
                )
                drs_data[team]["all"] = pd.concat(
                    [drs_data[team]["all"], drs_lap], ignore_index=True
                )

            if race not in data[team]:
                data[team][race] = lap
                drs_data[team][race] = drs_lap
            else:
                data[team][race] = pd.concat([data[team][race], lap], ignore_index=True)
                drs_data[team][race] = pd.concat(
                    [drs_data[team][race], drs_lap], ignore_index=True
                )


output_directory = "team_data"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Export non-DRS data
for team in data:
    output_folder = os.path.join(output_directory, team)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    data[team]["all"].to_csv(
        os.path.join(output_folder, "non_drs_data.csv"), index=False
    )

# Export DRS data
for team in drs_data:
    output_folder = os.path.join(output_directory, team)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    drs_data[team]["all"].to_csv(
        os.path.join(output_folder, "drs_data.csv"), index=False
    )

all_data = {
    "Alfa Romeo": {},
    "AlphaTauri": {},
    "Alpine": {},
    "Aston Martin": {},
    "Ferrari": {},
    "Haas": {},
    "McLaren": {},
    "Mercedes": {},
    "Williams": {},
    "Red Bull": {},
}

for team in data:
    # plt.scatter(data[team]['all']['v^2 (m^2/s^4)'], data[team]['all']['F_dissipated (N)'], label=team)
    regressions_data = []
    gradient_max = -1
    f_rr_max = -1
    gradient_min = 99999999999
    f_rr_min = 9999999999
    for race in data[team]:
        # Perform linear regression
        try:
            coeffs = np.polyfit(
                data[team][race]["v^2 (m^2/s^4)"],
                data[team][race]["F_dissipated (N)"],
                1,
            )
        except:
            print("skipped " + race + " for " + team + " no DRS")
            continue

        regression_line = np.poly1d(coeffs)

        # Calculate coefficients
        gradient = coeffs[0]
        f_rr = coeffs[1]
        c_rr = coeffs[1] / (4 * normal_force)

        if gradient > gradient_max:
            gradient_max = gradient
            all_data[team]["gradient_max"] = gradient
            all_data[team]["f_rr_max"] = f_rr
            all_data[team]["c_rr_max"] = c_rr
            all_data[team]["max_regression"] = regression_line

        if gradient < gradient_min:
            gradient_min = gradient
            all_data[team]["gradient_min"] = gradient
            all_data[team]["f_rr_min"] = f_rr
            all_data[team]["c_rr_min"] = c_rr
            all_data[team]["min_regression"] = regression_line

        if race == "all":
            all_data[team]["gradient"] = gradient
            all_data[team]["f_rr"] = f_rr
            all_data[team]["c_rr"] = c_rr
            all_data[team]["regression"] = regression_line

        # Append data to list
        regressions_data.append(
            {"Race": race, "gradient": gradient, "f_rr": f_rr, "c_rr": c_rr}
        )
        regressions_df = pd.DataFrame(regressions_data)
        regressions_df.to_csv(
            "team_data/" + team + "/regressions_data.csv", index=False
        )

for team in drs_data:
    # plt.scatter(data[team]['all']['v^2 (m^2/s^4)'], data[team]['all']['F_dissipated (N)'], label=team)
    drs_regressions_data = []
    drs_gradient_max = -1
    drs_f_rr_max = -1
    drs_gradient_min = 9999999999
    drs_f_rr_min = 999999999
    for race in drs_data[team]:
        # Perform linear regression

        try:
            drs_coeffs = np.polyfit(
                drs_data[team][race]["v^2 (m^2/s^4)"],
                drs_data[team][race]["F_dissipated (N)"],
                1,
            )
        except:
            print("skipped " + race + " for " + team + " DRS")
            continue

        drs_regression_line = np.poly1d(drs_coeffs)


        # Calculate coefficients

        drs_gradient = drs_coeffs[0]
        drs_f_rr = drs_coeffs[1]
        drs_c_rr = drs_f_rr / (4 * normal_force)

        if drs_gradient > drs_gradient_max:
            drs_gradient_max = drs_gradient
            all_data[team]["drs_gradient_max"] = drs_gradient
            all_data[team]["drs_f_rr_max"] = drs_f_rr
            all_data[team]["drs_c_rr_max"] = drs_c_rr
            all_data[team]["drs_max_regression"] = drs_regression_line

        if drs_gradient < drs_gradient_min:
            drs_gradient_min = drs_gradient
            all_data[team]["drs_gradient_min"] = drs_gradient
            all_data[team]["drs_f_rr_min"] = drs_f_rr
            all_data[team]["drs_c_rr_min"] = drs_c_rr
            all_data[team]["drs_min_regression"] = drs_regression_line

        if race == "all":
            all_data[team]["drs_gradient"] = drs_gradient
            all_data[team]["drs_f_rr"] = drs_f_rr
            all_data[team]["drs_c_rr"] = drs_c_rr
            all_data[team]["drs_regression"] = drs_regression_line

        # Append data to list
        drs_regressions_data.append(
            {"Race": race, "gradient": drs_gradient, "f_rr": drs_f_rr, "c_rr": drs_c_rr}
        )
        drs_regressions_df = pd.DataFrame(drs_regressions_data)
        drs_regressions_df.to_csv(
            "team_data/" + team + "/drs_regressions_data.csv", index=False
        )


# Create regression line for all data
x = []
y = []
for team in drs_data:
    x.extend(drs_data[team]["all"]["v^2 (m^2/s^4)"])
    y.extend(drs_data[team]["all"]["F_dissipated (N)"])

coeffs = np.polyfit(x, y, 1)
regression = np.poly1d(coeffs)

# Plot scatter and regression line
for team in drs_data:
    plt.scatter(
        drs_data[team]["all"]["v^2 (m^2/s^4)"],
        drs_data[team]["all"]["F_dissipated (N)"],
        label=team_to_car[team],
        alpha=0.5,
        s=10,
    )
plt.plot(x, regression(x), color="white", label=regression)

plt.xlabel("v^2 (m^2/s^4)")
plt.ylabel("F_dissipated (N)")
plt.title(
    "F_dissipated (N) vs v^2 (m^2/s^4) for Each Car with DRS for All Races",
    fontsize=10,
)
plt.legend()
plt.savefig("ALL-DRS.png", bbox_inches="tight")
plt.show()



x = []
y = []
x.extend(drs_data["Red Bull"]["all"]["v^2 (m^2/s^4)"])
y.extend(drs_data["Red Bull"]["all"]["F_dissipated (N)"])
sqrt_x = [np.sqrt(z) for z in x]

coeffs = np.polyfit(x, y, 1)
regression = np.poly1d(coeffs)

# Plot scatter and regression line
plt.scatter(
    drs_data["Red Bull"]["all"]["v^2 (m^2/s^4)"],
    drs_data["Red Bull"]["all"]["F_dissipated (N)"],
    label=team_to_car["Red Bull"],
    alpha=0.5,
    s=10,
)
plt.plot(x, regression(x), color="white", label=regression)

plt.xlabel("v^2 (m^2/s^4)")
plt.ylabel("F_dissipated (N)")
plt.title(
    "F_dissipated (N) vs v^2 (m^2/s^4) for the RB19 with DRS for All Races",
    fontsize=10,
)
plt.legend()
plt.savefig("RB-DRS.png", bbox_inches="tight")
plt.show()


x = []
y = []
x.extend(data["Red Bull"]["all"]["v^2 (m^2/s^4)"])
y.extend(data["Red Bull"]["all"]["F_dissipated (N)"])

coeffs = np.polyfit(x, y, 1)
regression = np.poly1d(coeffs)

# Plot scatter and regression line
plt.scatter(
    data["Red Bull"]["all"]["v^2 (m^2/s^4)"],
    data["Red Bull"]["all"]["F_dissipated (N)"],
    label=team_to_car["Red Bull"],
    alpha=0.5,
    s=10,
)
plt.plot(x, regression(x), color="white", label=regression)

plt.xlabel("v^2 (m^2/s^4)")
plt.ylabel("F_dissipated (N)")
plt.title(
    "F_dissipated (N) vs v^2 (m^2/s^4) for the RB19 without DRS for All Races",
    fontsize=10,
)
plt.legend()
plt.savefig("RB-NO-DRS.png", bbox_inches="tight")
plt.show()



x = []
y = []
for team in data:
    x.extend(data[team]["all"]["v^2 (m^2/s^4)"])
    y.extend(data[team]["all"]["F_dissipated (N)"])

coeffs = np.polyfit(x, y, 1)
regression = np.poly1d(coeffs)
# Plot scatter and regression line
for team in data:
    plt.scatter(
        data[team]["all"]["v^2 (m^2/s^4)"],
        data[team]["all"]["F_dissipated (N)"],
        label=team_to_car[team],
        alpha=0.5,
        s=10,
    )
plt.plot(x, regression(x), color="white", label=regression)

plt.xlabel("v^2 (m^2/s^4)")
plt.ylabel("F_dissipated (N)")
plt.title(
    "F_dissipated (N) vs v^2 (m^2/s^4) for Each Car without DRS for All Races",
    fontsize=10,
)
plt.legend()
plt.savefig("ALL-NO-DRS.png", bbox_inches="tight")
plt.show()


with open("all_data.csv", "w", newline="") as csvfile:
    fieldnames = [
        "rank",
        "car",
        "gradient",
        "gradient ±",
        "f_rr",
        "f_rr ±",
        "gradient (drs)",
        "gradient (drs) ±",
        "f_rr (drs)",
        "f_rr (drs) ±",
        "delta gradient",
        "delta gradient ±",
        "c_rr",
        "c_rr ±",
        "c_rr (drs)",
        "c_rr (drs) ±",
        "delta f_rr",
        "delta f_rr ±",
        "delta c_rr",
        "delta c_rr ±",
        "regression",
        "drs_regression",
        "drs_max_regression",
        "drs_min_regression",
        "max_regression",
        "min_regression",
        "v",
        "v max",
        "v min",
        "v ±",
        "v drs",
        "v drs max",
        "v drs min",
        "v drs ±",
        "delta v",
        "delta v ±",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write header
    writer.writeheader()

    # Write rows
    for team in all_data:
        writer.writerow(
            {
                "rank": team_to_rank[team],
                "car": team_to_car[team],
                "gradient": all_data[team]["gradient"],
                "gradient ±": abs((
                    all_data[team]["gradient_max"] - all_data[team]["gradient_min"]
                ))
                / 2,
                "f_rr": all_data[team]["f_rr"],
                "f_rr ±": abs((all_data[team]["f_rr_max"] - all_data[team]["f_rr_min"])) / 2,
                "gradient (drs)": all_data[team]["drs_gradient"],
                "gradient (drs) ±": (
                    all_data[team]["drs_gradient_max"]
                    - all_data[team]["drs_gradient_min"]
                )
                / 2,
                "f_rr (drs)": all_data[team]["drs_f_rr"],
                "f_rr (drs) ±": abs((
                    all_data[team]["drs_f_rr_max"] - all_data[team]["drs_f_rr_min"]
                ))
                / 2,
                "delta gradient": all_data[team]["gradient"]
                - all_data[team]["drs_gradient"],
                "delta gradient ±": abs((
                    (all_data[team]["gradient_max"] - all_data[team]["gradient_min"]))
                    / 2
                )
                + (
                    abs((
                        all_data[team]["drs_gradient_max"]
                        - all_data[team]["drs_gradient_min"]
                    ))
                    / 2
                ),
                "delta f_rr": all_data[team]["f_rr"] - all_data[team]["drs_f_rr"],
                "delta f_rr ±": abs((
                    (all_data[team]["f_rr_max"] - all_data[team]["f_rr_min"]) / 2
                ))
                + (
                    abs((all_data[team]["drs_f_rr_max"] - all_data[team]["drs_f_rr_min"]))
                    / 2
                ),
                "delta c_rr": all_data[team]["c_rr"] - all_data[team]["drs_c_rr"],
                "delta c_rr ±": (
                    abs((all_data[team]["c_rr_max"] - all_data[team]["c_rr_min"])) / 2
                )
                + (
                    abs((all_data[team]["drs_c_rr_max"] - all_data[team]["drs_c_rr_min"]))
                    / 2
                ),
                "regression": str(all_data[team]["regression"]),
                "drs_regression": str(all_data[team]["drs_regression"]),
                "drs_max_regression": str(all_data[team]["drs_max_regression"]),
                "drs_min_regression": str(all_data[team]["drs_min_regression"]),
                "max_regression": str(all_data[team]["max_regression"]),
                "min_regression": str(all_data[team]["min_regression"]),
                "c_rr": all_data[team]["c_rr"],
                "c_rr ±": abs((all_data[team]["c_rr_max"] - all_data[team]["c_rr_min"])) / 2,
                "c_rr (drs)": all_data[team]["drs_c_rr"],
                "c_rr (drs) ±": abs((
                    all_data[team]["drs_c_rr_max"] - all_data[team]["drs_c_rr_min"]
                ))/2,
                "v": fsolve(fn_to_solve, 100, args=all_data[team]["regression"])[0],
                "v max": fsolve(fn_to_solve, 100, args=all_data[team]["max_regression"])[0],
                "v min": fsolve(fn_to_solve, 100, args=all_data[team]["min_regression"])[0],
                "v drs": fsolve(fn_to_solve, 100, args=all_data[team]["drs_regression"])[0],
                "v drs max": fsolve(fn_to_solve, 100, args=all_data[team]["drs_max_regression"])[0],
                "v drs min": fsolve(fn_to_solve, 100, args=all_data[team]["drs_min_regression"])[0],
                "delta v": fsolve(fn_to_solve, 100, args=all_data[team]["drs_regression"])[0] - fsolve(fn_to_solve, 100, args=all_data[team]["regression"])[0],
                "delta v ±": (abs(
                    fsolve(fn_to_solve, 100, args=all_data[team]["max_regression"])[0] - fsolve(fn_to_solve, 100, args=all_data[team]["min_regression"])[0]
                )/2) + 
                (abs(
                    fsolve(fn_to_solve, 100, args=all_data[team]["drs_max_regression"])[0] - fsolve(fn_to_solve, 100, args=all_data[team]["drs_min_regression"])[0]
                )/2),
                "v ±": abs(
                    fsolve(fn_to_solve, 100, args=all_data[team]["max_regression"])[0] - fsolve(fn_to_solve, 100, args=all_data[team]["min_regression"])[0]
                )/2,
                "v drs ±": abs(
                    fsolve(fn_to_solve, 100, args=all_data[team]["drs_max_regression"])[0] - fsolve(fn_to_solve, 100, args=all_data[team]["drs_min_regression"])[0]
                )/2
            }
        )

# create regression for redbull max and min
plt.plot(
    x,
    all_data["Red Bull"]["regression"](x),
    color="white",
    label="F_dissipated " + str(all_data["Red Bull"]["regression"]),
)
plt.plot(
    x,
    all_data["Red Bull"]["max_regression"](x),
    color="red",
    label="Max F_dissipated " + str(all_data["Red Bull"]["max_regression"]),
)
plt.plot(
    x,
    all_data["Red Bull"]["min_regression"](x),
    color="blue",
    label="Min F_dissipated " + str(all_data["Red Bull"]["min_regression"]),
)
plt.xlabel("v^2 (m^2/s^4)")
plt.ylabel("F_dissipated (N)")
plt.title(
    "F_dissipated (N) vs v^2 (m^2/s^4) for the RB19 without DRS",
    fontsize=10,
)
plt.legend()
plt.savefig("RedBull-MAX-MIN-NO-DRS.png", bbox_inches="tight")
plt.show()

plt.plot(
    x,
    all_data["Red Bull"]["drs_regression"](x),
    color="white",
    label="F_dissipated " + str(all_data["Red Bull"]["drs_regression"]),
)
plt.plot(
    x,
    all_data["Red Bull"]["drs_max_regression"](x),
    color="red",
    label="Max F_dissipated " + str(all_data["Red Bull"]["drs_max_regression"]),
)
plt.plot(
    x,
    all_data["Red Bull"]["drs_min_regression"](x),
    color="blue",
    label="Min F_dissipated " + str(all_data["Red Bull"]["drs_min_regression"]),
)
plt.xlabel("v^2 (m^2/s^4)")
plt.ylabel("F_dissipated (N)")
plt.title(
    "F_dissipated (N) vs v^2 (m^2/s^4) for the RB19 with DRS",
    fontsize=10,
)
plt.legend()
plt.savefig("RedBull-MAX-MIN-DRS.png", bbox_inches="tight")
plt.show()




new_regression = all_data["Red Bull"]["regression"].c

plt.plot(
    sqrt_x,
    np.poly1d([new_regression[0], 0, new_regression[1]])(sqrt_x),
    color="white",
    label=("F_dissipated " + str(all_data["Red Bull"]["drs_regression"])).replace("x", "x^2")
)
plt.plot(
    sqrt_x,
    [max_engine_power/(z) for z in sqrt_x],
    color="white",
    label=("F_dissipated " + str(all_data["Red Bull"]["drs_regression"])).replace("x", "x^2")
)
plt.xlabel("v (m/s)")
plt.ylabel("F_dissipated (N)")
plt.title(
    "aaa",
    fontsize=10,
)
plt.legend()
plt.savefig("test.png", bbox_inches="tight")
plt.show()

