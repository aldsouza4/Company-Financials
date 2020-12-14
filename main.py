import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
global percentage


def clean_values(num):
    if "," in num and "%" not in num:
        num = num.replace(",", "")
        return float(num)
    if "," and "%" in num:
        num = num[:-1]
        num = num.replace(",", "")
        return float(num)
    if num[0] == "(":
        return float(num[1:-1])
    if num[-1] == "%":
        temp = float(num[:-1])
        return temp
    else:
        return float(num)

def naming_function(name_):
    if name_ == "quarter_revenue":
        return "Quarter Revenue"
    elif name_ == "quarter_operating_profits":
        return "Quarter Operating Profits"
    elif name_ == "quarter_operating_margins":
        return "Quarter Operating Margins(%)"
    elif name_ == "quarter_profit_before_tax":
        return "Quarter Profit Before Tax"
    elif name_ == "quarter_net_profit":
        return "Quarter Net Profit"

    elif name_ == "annual_revenue":
        return "Annual Revenue"
    elif name_ == "annual_operating_profits":
        return "Annual Operating Profits"
    elif name_ == "annual_operating_margins":
        return "Annual Operating Margins(%)"
    elif name_ == "annual_profit_before_tax":
        return "Annual Profit Before Tax"
    elif name_ == "annual_net_profit":
        return "Annual Net Profit"
    elif name_ == "annual_eps":
        return "Annual EPS"
    else:
        return "Something Broke"

def clean_ser(val):
    return (np.around((val.reshape(-1)), 2)).tolist()


def make_predictions(input_list, column_list, num_terms_pred, name):

    pd_index = naming_function(name_=name)
    lm = LinearRegression()
    index_input_list = np.arange(len(input_list))

    input_list = np.array(input_list).reshape(-1, 1)

    index_input_list = np.array(index_input_list).reshape(-1, 1)

    lm.fit(X=index_input_list, y=input_list)

    predict_me = np.arange(len(input_list), len(input_list) + num_terms_pred)

    predict_me = predict_me.reshape(-1, 1)
    predictions = lm.predict(predict_me)
    predictions = clean_ser(predictions)
    predict_me = clean_ser(predict_me)

    index_input_list = clean_ser(index_input_list)
    input_list = clean_ser(input_list)

    predict_label = np.arange(len(input_list), len(input_list) + num_terms_pred)
    predict_label = clean_ser(predict_label)

    cont_predictions = predictions

    predict_label.append(index_input_list[-1])
    cont_predictions.append(input_list[-1])

    # return predictions

    scrap = int(column_list[-1][-2:]) + 1
    for i in range(num_terms_pred):
        column_list.append("March 20{} E".format(scrap + i))

    plt.figure(figsize=(15, 8))
    sns.lineplot(x=index_input_list, y=input_list, color='red')
    sns.lineplot(x=predict_label, y=cont_predictions, color='green', linewidth=2)
    plt.xticks(ticks=range(0, len(column_list)), labels=column_list, rotation='vertical', fontsize=8)
    plt.xlabel('Results declared in', fontsize=14)
    if percentage:
        plt.ylabel('In %', fontsize=14)
        plt.title("{}".format(pd_index), fontsize=18)
        plt.show()

    else:
        plt.ylabel('INR in crores', fontsize=14)
        plt.title("{}".format(pd_index), fontsize=18)
        plt.show()

    return input_list + predictions


class FinancialData(object):

    def __init__(self):
        self.df = pd.read_html("https://www.screener.in/company/STLTECH/consolidated/")

    def financial_results(self, name, display=False):

        data = None

        if name == "quarter":
            table_index = 0
            data = pd.DataFrame(self.df[table_index])

        elif name == "annual":
            table_index = 1
            data = pd.DataFrame(self.df[table_index])
            data.drop("TTM", axis=True, inplace=True)


        data.dropna(axis=0, inplace=True)

        column_list = data.columns
        column_list = column_list[1:]
        column_list = column_list.tolist()


        for i in column_list:
            data[i] = data[i].apply(clean_values)

        if display:
            return data.to_string()
        else:
            return data

    def get_results(self, table_name, name, as_list=False, plot_inner=False, asDataFrame=True):
        data = self.financial_results(name=table_name)
        percentage = False
        column_list = data.columns
        column_list = column_list[1:]
        column_list = column_list.tolist()

        if name == "quarter_revenue":
            row_index = 0
        elif name == "quarter_operating_profits":
            row_index = 2
        elif name == "quarter_operating_margins":
            row_index = 3
            percentage = True
        elif name == "quarter_profit_before_tax":
            row_index = 7
        elif name == "quarter_net_profit":
            row_index = 9

        elif name == "annual_revenue":
            row_index = 0
        elif name == "annual_operating_profits":
            row_index = 2
        elif name == "annual_operating_margins":
            row_index = 3
            percentage = True
        elif name == "annual_profit_before_tax":
            row_index = 7
        elif name == "annual_net_profit":
            row_index = 9
        elif name == "annual_eps":
            row_index = 10
        else:
            return "Something Broke"

        data = list(data.iloc[row_index])
        data = data[1:]

        if plot_inner:
            index_list = []
            for i in range(len(data)):
                index_list.append(i)

            pd_index = naming_function(name_=name)


            plt.figure(figsize=(15, 8))
            sns.lineplot(x=index_list, y=data)
            plt.xticks(ticks=range(0, len(index_list)), labels=column_list, rotation='vertical', fontsize=8)
            plt.xlabel('Results declared in', fontsize=14)
            if percentage:
                plt.ylabel('In %', fontsize=14)
                plt.title("{}".format(pd_index), fontsize=18)
                plt.show()

            else:
                plt.ylabel('INR in crores', fontsize=14)
                plt.title("{}".format(pd_index), fontsize=18)
                plt.show()

        if as_list:
            return data

        if asDataFrame:
            pd_index = naming_function(name_=name)

            data = pd.DataFrame(data=[data], columns=column_list, index=[pd_index])
            return data

    # -----------QUARTER DATA-----------------------------------------------------

    def quarter_revenue(self, as_list=False, plot=False, as_DataFrame=True):
        r_name = "quarter_revenue"
        t_name = "quarter"
        return self.get_results(table_name=t_name, name=r_name, as_list=as_list, plot_inner=plot,
                                asDataFrame=as_DataFrame)

    def quarter_operating_profits(self, as_list=False, plot=False, as_DataFrame=True):
        r_name = "quarter_operating_profits"
        t_name = "quarter"
        return self.get_results(table_name=t_name, name=r_name, as_list=as_list, plot_inner=plot,
                                asDataFrame=as_DataFrame)

    def quarter_operating_margins(self, as_list=False, plot=False, as_DataFrame=True):
        r_name = "quarter_operating_margins"
        t_name = "quarter"
        return self.get_results(table_name=t_name, name=r_name, as_list=as_list, plot_inner=plot,
                                asDataFrame=as_DataFrame)

    def quarter_profit_before_tax(self, as_list=False, plot=False, as_DataFrame=True):
        r_name = "quarter_profit_before_tax"
        t_name = "quarter"
        return self.get_results(table_name=t_name, name=r_name, as_list=as_list, plot_inner=plot,
                                asDataFrame=as_DataFrame)

    def quarter_net_profit(self, as_list=False, plot=False, as_DataFrame=True):
        r_name = "quarter_net_profit"
        t_name = "quarter"
        return self.get_results(table_name=t_name, name=r_name, as_list=as_list, plot_inner=plot,
                                asDataFrame=as_DataFrame)

    # -----------ANNUAL DATA-----------------------------------------------------

    def annual_revenue(self, as_list=False, plot=False, as_DataFrame=True):
        r_name = "annual_revenue"
        t_name = "annual"
        return self.get_results(table_name=t_name, name=r_name, as_list=as_list, plot_inner=plot,
                                asDataFrame=as_DataFrame)

    def annual_operating_profits(self, as_list=False, plot=False, as_DataFrame=True):
        r_name = "annual_operating_profits"
        t_name = "annual"
        return self.get_results(table_name=t_name, name=r_name, as_list=as_list, plot_inner=plot,
                                asDataFrame=as_DataFrame)

    def annual_operating_margins(self, as_list=False, plot=False, as_DataFrame=True):
        r_name = "annual_operating_margins"
        t_name = "annual"
        return self.get_results(table_name=t_name, name=r_name, as_list=as_list, plot_inner=plot,
                                asDataFrame=as_DataFrame)

    def annual_profit_before_tax(self, as_list=False, plot=False, as_DataFrame=True):
        r_name = "annual_profit_before_tax"
        t_name = "annual"
        return self.get_results(table_name=t_name, name=r_name, as_list=as_list, plot_inner=plot,
                                asDataFrame=as_DataFrame)

    def annual_net_profit(self, as_list=False, plot=False, as_DataFrame=True):
        r_name = "annual_net_profit"
        t_name = "annual"
        return self.get_results(table_name=t_name, name=r_name, as_list=as_list, plot_inner=plot,
                                asDataFrame=as_DataFrame)

    def annual_eps(self, as_list=False, plot=False, as_DataFrame=True):
        r_name = "annual_eps"
        t_name = "annual"
        return self.get_results(table_name=t_name, name=r_name, as_list=as_list, plot_inner=plot,
                                asDataFrame=as_DataFrame)


t = FinancialData()
print(t.annual_operating_margins(plot=True))

