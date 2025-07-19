import pandas as pd
import matplotlib.pyplot as plt


def main():
    file_path = 'train.csv'
    data = pd.read_csv(file_path)

    # df = pd.DataFrame()
    # df["Name"] = ["danil", "pasha", "tagir", "kirill"]
    # df["Age"] = [10, 20, 30, 40]
    # df.index = range(1,5)
    # print(df)
    # print(df.iloc[2])
    # print(df.loc[2])

    # plt.scatter(range(0, len(data)), data.Age)
    # plt.show()

    # manOver19 = data[(data.Sex == "male") & (data.Age >= 19)]
    # print(manOver19)

    # Мой вариант
    female = data[data.Sex == "female"]
    femaleClass = female["Pclass"].value_counts()

    male = data[data.Sex == "male"]
    maleClass = male["Pclass"].value_counts()

    # Вариант преподавателя
    groupData = data.groupby(["Sex", "Pclass"]).size()
    groupData.plot(kind="bar")
    plt.show()


if __name__ == '__main__':
    main()
