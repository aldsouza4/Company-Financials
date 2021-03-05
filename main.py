import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas_datareader as wb
import numpy as np
from statistics import mean
from datetime import datetime
from dateutil.relativedelta import relativedelta


def clean_values(num):
    """
    takes in a string and returns the int or float or percentage (in decimal)
    :param num:str
    :return:
    """
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
    """
    takes in the type of result nd returns the result in presentable form
    :param name_:str
    :return:
    """
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
    elif name_ == "share_capital":
        return "Share Capital"
    elif name_ == "reserves":
        return "Reserves"
    elif name_ == "borrowings":
        return "Borrowing"
    elif name_ == "other_liabilities":
        return "Other Liabilities"
    elif name_ == "total_liabilities":
        return "Total Liabilities"
    elif name_ == "fixed_assets":
        return "Fixed Assets"
    elif name_ == "capital_work":
        return "CWIP"
    elif name_ == "investments":
        return "Investments"
    elif name_ == "other_assets":
        return "Other Assets"
    elif name_ == "total_assets":
        return "Total Assets"
    elif name_ == "cash_flow_operating_activities":
        return "Cash from Operating Activities"
    elif name_ == "cash_flow_investing_activities":
        return "Cash from Investing Activities"
    elif name_ == "cash_flow_financing_activities":
        return "Cash from Financing Activities"
    elif name_ == "net_cash_flow":
        return "Net Cash Flow"
    elif name_ == "roce":
        return "ROCE (%)"
    elif name_ == "net_income_margins":
        return "Net Income Margins (%)"
    elif name_ == "operating_to_net_income":
        return "Operating to Net_income (%)"
    elif name_ == "free_cash_flow":
        return "Free Cash Flow"
    elif name_ == "freecash_to_net_income":
        return "Free Cash Flow to Net Income"
    else:
        return "Something Broke"


def clean_ser(val):
    # takes the vale, reshapes it and returns it in list format
    return (np.around((val.reshape(-1)), 2)).tolist()


class FinancialData(object):
    """
    Financial data of the company
    """

    def __init__(self, ticker):
        """
        :param ticker:takes in the ticker of the comapny
        """
        self.ticker = ticker.upper()
        self.df = pd.read_html("https://www.screener.in/company/{}/consolidated/".format(self.ticker))

    def financial_results(self, name, display=False):
        """
        takes in  the name of the table and returns the table
        included tables are:
        1.Quarter Profit and Loss Statements
        2.Annual Profit and loss Statements
        3.Balance Sheet Items
        4.Cash Flow items

        :param self:
        :param name: takes in the table name
        :param display: True to display the tabel
        :return: returns the table
        """

        data = None

        if name == "quarter":
            table_index = 0
            data = pd.DataFrame(self.df[table_index])

        elif name == "annual":
            table_index = 1
            data = pd.DataFrame(self.df[table_index])
            data.drop("TTM", axis=True, inplace=True)

        elif name == "balance":
            table_index = 6
            data = pd.DataFrame(self.df[table_index])
            data.dropna(axis=0, inplace=True)
            self.column_list = data.columns
            self.column_list = self.column_list[1:]
            self.column_list = self.column_list.tolist()
            if display:
                return data.to_string()
            else:
                return data
        elif name == "cash_flow":
            table_index = 7
            data = pd.DataFrame(self.df[table_index])
            data.dropna(axis=0, inplace=True)
            self.column_list = data.columns
            self.column_list = self.column_list[1:]
            self.column_list = self.column_list.tolist()
            if display:
                return data.to_string()
            else:
                return data

        elif name == "ROCE":
            table_index = 8
            data = pd.DataFrame(self.df[table_index])
            data.dropna(axis=1, inplace=True)

        data.dropna(axis=0, inplace=True)

        self.column_list = data.columns
        self.column_list = self.column_list[1:]
        self.column_list = self.column_list.tolist()

        for i in self.column_list:
            data[i] = data[i].apply(clean_values)

        if display:
            return data.to_string()
        else:
            return data

    def __get_results(self, table_name, name, as_list=False, plot_inner=False, asDataFrame=True):
        """
        private method
        takes in the table item and returns it in the desired form

        :param table_name:takes in the name of the table
        :param name: name of the item from the list
        :param as_list: if True returns the the row as a list
        :param plot_inner: if True Plots a seaborn line plot
        :param asDataFrame: if True returns the the row as a Dataframe
        :return: returns the row in desired data type

        """
        data = self.financial_results(name=table_name)
        self.percentage = False

        if name == "quarter_revenue":
            row_index = 0
        elif name == "quarter_operating_profits":
            row_index = 2
        elif name == "quarter_operating_margins":
            row_index = 3
            self.percentage = True
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
            self.percentage = True
        elif name == "annual_profit_before_tax":
            row_index = 7
        elif name == "annual_net_profit":
            row_index = 9
        elif name == "annual_eps":
            row_index = 10

        elif name == "share_capital":
            row_index = 0
        elif name == "reserves":
            row_index = 1
        elif name == "borrowings":
            row_index = 2
        elif name == "other_liabilities":
            row_index = 3
        elif name == "total_liabilities":
            row_index = 4
        elif name == "fixed_assets":
            row_index = 5

        elif name == "capital_work":
            row_index = 6
        elif name == "investments":
            row_index = 7
        elif name == "other_assets":
            row_index = 8
        elif name == "total_assets":
            row_index = 9

        elif name == "cash_flow_operating_activities":
            row_index = 0
        elif name == "cash_flow_investing_activities":
            row_index = 1
        elif name == "cash_flow_financing_activities":
            row_index = 2
        elif name == "net_cash_flow":
            row_index = 3
        elif name == "roce":
            row_index = 0
            self.percentage = True
        else:
            return "Something Broke"

        data = list(data.iloc[row_index])
        data = data[1:]
        self.pd_index = naming_function(name_=name)

        if plot_inner:
            self.index_list = []
            for i in range(len(data)):
                self.index_list.append(i)

            plt.figure(figsize=(15, 8))
            sns.set_style("darkgrid")
            sns.lineplot(x=self.index_list, y=data, linewidth=2, marker="o")
            plt.xlabel('Results declared in', fontsize=14)
            plt.xticks(ticks=range(0, len(self.index_list)), labels=self.column_list, rotation='vertical', fontsize=8)
            sns.set(style='dark')

            if self.percentage:
                plt.ylabel('In %', fontsize=14)
            else:
                plt.ylabel('INR in crores', fontsize=14)

            plt.title("{}".format(self.pd_index), fontsize=18)
            plt.legend(labels=['Declared'])
            plt.tight_layout()
            plt.plot(marker="o")
            plt.show()

        if as_list:
            return data

        if asDataFrame:
            data = pd.DataFrame(data=[data], columns=self.column_list, index=[self.pd_index])
            return data

    def make_predictions(self, input_list, num_terms_pred=3, plot=False, as_list=False, asDataFrame=True,
                         only_prediction_list=False):
        """

        :param input_list: Takes in the input as list
        :param num_terms_pred: takes in the number of year to be predicted : default is 3 years
        :param plot: If True plots a seaborn line plot
        :param as_list: if True returns  in list form
        :param asDataFrame: if True returns in DataFrame
        :param only_prediction_list: if True, returns only predicted list
        :return:
        """

        name = self.r_name

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

        index_input_list = clean_ser(index_input_list)
        input_list = clean_ser(input_list)

        predict_label = np.arange(len(input_list), len(input_list) + num_terms_pred)
        predict_label = clean_ser(predict_label)

        cont_predictions = predictions

        predict_label.append(index_input_list[-1])
        cont_predictions.append(input_list[-1])

        scrap = int(self.column_list[-1][-2:]) + 1
        for i in range(num_terms_pred):
            self.column_list.append("March 20{} E".format(scrap + i))

        if as_list:
            return input_list + predictions[:-1]

        if plot:
            plt.figure(figsize=(15, 8))
            sns.set_theme()
            sns.lineplot(x=index_input_list, y=input_list, color='red', linewidth=2, marker="o")
            sns.lineplot(x=predict_label, y=cont_predictions, color='green', linewidth=3, marker="o")
            plt.xticks(ticks=range(0, len(self.column_list)), labels=self.column_list, rotation='vertical', fontsize=8)
            plt.xlabel('Results declared in', fontsize=14)
            sns.set(style='dark')

            if self.percentage:
                plt.ylabel('In %', fontsize=14)

            else:
                plt.ylabel('INR in crores', fontsize=14)

            plt.title("{}".format(pd_index), fontsize=18)
            plt.legend(labels=['Declared', 'Expected'])
            plt.tight_layout()
            plt.show()

        if only_prediction_list:
            return predictions[-num_terms_pred:]

        if asDataFrame:
            data = pd.DataFrame(data=[input_list + predictions[:-1]], columns=self.column_list, index=[self.pd_index])
            return data

    def disp_data(self, data, as_list=False, plot=False, as_DataFrame=True, average=False):
        """

        :param data: takes in the data
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :param average: if True returns average of the Net income Margins
        :return: returns the output in desired form/ Object
        """
        self.pd_index = naming_function(self.r_name)

        if as_list:
            data = [round(num, 2) for num in data]
            return data
        elif average:
            return clean_ser(mean(data))[0]
        elif plot:
            self.index_list = []
            for i in range(len(data)):
                self.index_list.append(i)

            plt.figure(figsize=(15, 8))
            sns.set_style("darkgrid")
            sns.lineplot(x=self.index_list, y=data, linewidth=2, marker="o")
            plt.xlabel('Results declared in', fontsize=14)
            plt.xticks(ticks=range(0, len(self.index_list)), labels=self.column_list, rotation='vertical', fontsize=8)
            sns.set(style='dark')

            if self.percentage:
                plt.ylabel('In %', fontsize=14)
            else:
                plt.ylabel('INR in crores', fontsize=14)

            plt.title("{}".format(self.pd_index), fontsize=18)
            plt.legend(labels=['Declared'])
            plt.tight_layout()
            plt.plot(marker="o")
            plt.show()

        elif as_DataFrame:
            data = pd.DataFrame(data=[data], columns=self.column_list, index=[self.pd_index])
            return data

        elif average:
            return mean(data)

    # -----------QUARTER DATA-----------------------------------------------------

    def quarter_revenue(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "quarter_revenue"
        t_name = "quarter"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def quarter_operating_profits(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "quarter_operating_profits"
        t_name = "quarter"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def quarter_operating_margins(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "quarter_operating_margins"
        t_name = "quarter"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def quarter_profit_before_tax(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "quarter_profit_before_tax"
        t_name = "quarter"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def quarter_net_profit(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "quarter_net_profit"
        t_name = "quarter"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    # -----------ANNUAL DATA-----------------------------------------------------

    def annual_revenue(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "annual_revenue"
        t_name = "annual"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def annual_operating_profits(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "annual_operating_profits"
        t_name = "annual"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def annual_operating_margins(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "annual_operating_margins"
        t_name = "annual"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def annual_profit_before_tax(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "annual_profit_before_tax"
        t_name = "annual"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def annual_net_profit(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "annual_net_profit"
        t_name = "annual"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def annual_eps(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "annual_eps"
        t_name = "annual"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    # ------------Balance Sheet Items--------------------------------

    def share_capital(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "share_capital"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def reserves(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "reserves"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def borrowings(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "borrowings"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def other_liabilities(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "other_liabilities"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def total_liabilities(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "total_liabilities"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def fixed_assets(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "fixed_assets"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def capital_work(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "capital_work"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def other_assets(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "other_assets"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def total_assets(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "total_assets"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    # ------------Cash Flow Items ------------------------------------

    def cash_from_operating_activity(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "cash_flow_operating_activities"
        t_name = "cash_flow"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def cash_flow_investing_activities(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "cash_flow_investing_activities"
        t_name = "cash_flow"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def cash_from_financing_activity(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "cash_flow_financing_activities"
        t_name = "cash_flow"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def net_cash_flow(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "net_cash_flow"
        t_name = "cash_flow"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def roce(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from __get_results
        """

        self.r_name = "roce"
        t_name = "ROCE"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def free_cash_flow(self, as_list=False, plot=False, as_DataFrame=True):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :return: returns the output from disp_data method
        """

        fcf = [x + y for x, y in zip(self.cash_from_operating_activity(as_list=True),
                                     self.cash_flow_investing_activities(as_list=True))]
        self.r_name = "free_cash_flow"
        return self.disp_data(data=fcf, as_list=as_list, plot=plot, as_DataFrame=as_DataFrame, average=False)

    def net_income_margins(self, as_list=False, plot=False, as_DataFrame=True, average=False):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :param average: if True returns average of the Net income Margins
        :return: returns the output from disp_data method
        """

        net_income = self.annual_net_profit(as_list=True)
        revenue = self.annual_revenue(as_list=True)
        nim = [x / y for x, y in zip(net_income, revenue)]
        nim = [element * 100 for element in nim]
        self.percentage = True
        self.r_name = 'net_income_margins'
        return self.disp_data(data=nim, as_list=as_list, plot=plot, as_DataFrame=as_DataFrame, average=average)

    def operating_to_net_income(self, as_list=False, plot=False, as_DataFrame=True, average=False):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :param average: if True returns average of the Net income Margins
        :return: returns the output from disp_data method
        """

        net_income = self.annual_net_profit(as_list=True)
        operating_income = self.cash_from_operating_activity(as_list=True)
        ratio_list = [y / x for x, y in zip(net_income, operating_income)]
        ratio_list = [element * 100 for element in ratio_list]
        self.percentage = True
        self.r_name = 'operating_to_net_income'
        return self.disp_data(data=ratio_list, as_list=as_list, plot=plot, as_DataFrame=as_DataFrame, average=average)

    def free_cash_to_net_income(self, as_list=False, plot=False, as_DataFrame=True, average=False):
        """
        :param as_list: if True returns as list
        :param plot: if True returns a seaborn line plot
        :param as_DataFrame: if True returns a Data frame
        :param average: if True returns average of the Net income Margins
        :return: returns the output from disp_data method
        """
        net_income = self.annual_net_profit(as_list=True)
        free_cash_flow = self.free_cash_flow(as_list=True)
        ratio_list = [y / x for x, y in zip(net_income, free_cash_flow)]
        # ratio_list = [element * 100 for element in ratio_list]
        self.percentage = True
        self.r_name = 'freecash_to_net_income'
        return self.disp_data(data=ratio_list, as_list=as_list, plot=plot, as_DataFrame=as_DataFrame, average=average)

    def shares_outstanding(self):
        """
        :return: Returns number of outstanding shares ( Data from Yahoo Finance )
        """
        data = pd.read_html("https://in.finance.yahoo.com/quote/{0}.NS/key-statistics?p={0}.NS&.tsrc=fin-srch"
                            .format(self.ticker))
        raw_shares = data[2].iloc[2][1]

        if raw_shares[-1] == "B":
            self.shares = float(raw_shares[:-1]) * 1000000000
        elif raw_shares[-1] == "M":
            self.shares = float(raw_shares[:-1]) * 1000000
        else:
            self.shares = float(raw_shares) * 1000

        return self.shares

    def discounted_cash_flow(self, net_profit=True, operating_cash=False, free_cash=False, required_rate_of_return=0.12,
                             perpetual_growth_rate=0.04, num_yrs_in_rrr=4, using_to_predict=False, p_net_profit=True,
                             p_operating_cash=False, p_free_cash=False, p_yrs=1):
        """
        Discounted cash flow (DCF) is a valuation method used to estimate the value of an investment based on its
        expected future cash flows. DCF analysis attempts to figure out the value of an investment today,
        based on projections of how much money it will generate in the future.

        :param perpetual_growth_rate: Growth rate of the company after the RRR period : defaulted to 4 %
        :param required_rate_of_return: Required rate of return : Defaulted to 12%
        :param net_profit: If True uses Net Profit for analysis
        :param operating_cash: If True uses operating cash for analysis
        :param free_cash:If True uses free cash flow for analysis
        :param num_yrs_in_rrr: Number of years in Required rate of return
        :param using_to_predict: For discounted_cash_flow_price_predictor method
        :param p_net_profit: For discounted_cash_flow_price_predictor method
        :param p_operating_cash: For discounted_cash_flow_price_predictor method
        :param p_free_cash: For discounted_cash_flow_price_predictor method
        :param p_yrs: For discounted_cash_flow_price_predictor method
        :return: returns Fair value of the share
        """

        if using_to_predict:
            net_revenue = self.annual_revenue(as_list=True)
            rev_predict = self.make_predictions(net_revenue, as_list=True, num_terms_pred=p_yrs + num_yrs_in_rrr)

        elif using_to_predict == False:
            rev_predict = self.make_predictions(self.annual_revenue(as_list=True), num_terms_pred=num_yrs_in_rrr,
                                                as_list=True)
        else:
            rev_predict = 0

        works = True
        nim_avg = self.net_income_margins(average=True) / 100
        net_income_predict = []
        for i in range(1, num_yrs_in_rrr + 1):
            temp = (rev_predict[-i] * nim_avg)
            net_income_predict.append(temp)
        net_income_predict = [round(num, 2) for num in net_income_predict]
        net_income_predict.reverse()
        income_predict = []

        if operating_cash or (using_to_predict and p_operating_cash):
            opm_to_nim_avg = self.operating_to_net_income(average=True)
            # if opm_to_nim_avg is negative return not possible
            if opm_to_nim_avg < 0:
                print("Cannot be determined with Operating cash flow ")
                print("continuing using net profit")
                income_predict = net_income_predict
                works = False
            #     continue using net income
            if works:
                if abs(opm_to_nim_avg) > 1:
                    opm_to_nim_avg = opm_to_nim_avg / 100
                for i in range(1, 5):
                    temp = (net_income_predict[-i] * opm_to_nim_avg)
                    income_predict.append(temp)
                    income_predict = [round(num, 2) for num in income_predict]
                    income_predict.reverse()

        elif free_cash or (using_to_predict and p_free_cash):
            fcf_to_ni_avg = self.free_cash_to_net_income(average=True)
            # if fcf_to_ni_avg is negative return not possible
            if fcf_to_ni_avg < 0:
                print("Cannot be determined with free cash flow ")
                print("continuing using net profit")
                income_predict = net_income_predict
                works = False

            if works:
                if fcf_to_ni_avg > 1:
                    fcf_to_ni_avg = fcf_to_ni_avg / 100

                for i in range(1, 5):
                    temp = (net_income_predict[-i] * fcf_to_ni_avg)
                    income_predict.append(temp)
                    income_predict = [round(num, 2) for num in income_predict]
                    income_predict.reverse()

        elif net_profit or (using_to_predict and p_net_profit):
            income_predict = net_income_predict

        discount_rate = required_rate_of_return
        shares_outstanding = self.shares_outstanding()

        terminal_value = (income_predict[-1] * (1 + perpetual_growth_rate)) / (
                required_rate_of_return - perpetual_growth_rate)

        #       Applying Discount Factor

        discount_factor_list = []
        for i in range(1, 5):
            discount_factor_list.append((1 + discount_rate) ** i)

        present_value_future_cashflow = [x / y for x, y in zip(income_predict, discount_factor_list)]
        present_value_future_cashflow.append(terminal_value)
        todays_value_futurecash = sum(present_value_future_cashflow)

        value_of_share = todays_value_futurecash / (shares_outstanding / 10000000)
        value_of_share = round(value_of_share, 3)

        return value_of_share
        # print(value_of_share, " value per share ")

    def discounted_cash_flow_price_predictor(self, predict_no_yrs=1, plot=False, net_profit=True, operating_cash=False,
                                             free_cash=False, required_rate_of_return=0.12, perpetual_growth_rate=0.04):
        """
        :param perpetual_growth_rate: Growth rate of the company after the RRR period : defaulted to 4 %
        :param required_rate_of_return: Required rate of return : Defaulted to 12%
        :param predict_no_yrs: Stock price to be predicted of _ number of years
        :param net_profit: If True uses Net Profit for analysis
        :param operating_cash: If True uses operating cash for analysis
        :param free_cash:If True uses free cash flow for analysis
        :return: resturns the fair vale of the stock _ number of years later
        """

        stock_price = self.discounted_cash_flow(using_to_predict=True, p_yrs=predict_no_yrs, p_net_profit=net_profit,
                                                p_operating_cash=operating_cash, p_free_cash=free_cash,
                                                required_rate_of_return=required_rate_of_return,
                                                perpetual_growth_rate=perpetual_growth_rate)
        if plot:
            start_date = datetime.now() - relativedelta(years=5)
            self.start_input = "{0}-{1}-{2}".format(start_date.year, start_date.month, start_date.day)

            tick = "{}.NS".format(self.ticker)
            stock_price_data = wb.DataReader(tick, data_source='yahoo', start=self.start_input)['Adj Close']
            stock_price_data = stock_price_data.to_frame()
            p_date = datetime.now() + relativedelta(years=predict_no_yrs)
            stock_price_data = stock_price_data.reset_index()
            stock_price_data['Date'] = stock_price_data['Date'].apply(lambda x: x.date())
            predict_data = pd.DataFrame(columns=['Date', 'Adj Close'])
            predict_data.loc[0] = [stock_price_data.iloc[-1]['Date'], stock_price_data.iloc[-1]['Adj Close']]
            predict_data.loc[1] = [p_date.date(), stock_price]

            plt.figure(figsize=(15, 8))
            sns.set_style("darkgrid")
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.title("{}".format(tick), fontsize=18)
            plt.tight_layout()
            plt.plot_date(x=predict_data['Date'], y=predict_data['Adj Close'], linestyle='solid', marker=None)
            plt.plot_date(x=stock_price_data['Date'], y=stock_price_data['Adj Close'], linestyle='solid', marker=None)
            plt.legend(labels=["Expectation", 'Declared'])
            plt.show()
        return stock_price

    def beta(self, num_years=5):
        tick = "{}.NS".format(self.ticker)
        tickers = [tick, '^NSEI']
        b_data = pd.DataFrame()
        start_date = datetime.now() - relativedelta(years=num_years)
        self.start_input = "{0}-{1}-{2}".format(start_date.year, start_date.month, start_date.day)

        for st in tickers:
            b_data[st] = wb.DataReader(st, data_source='yahoo', start=self.start_input)['Adj Close']

        sec_returns = np.log(b_data / b_data.shift(1))
        cov = sec_returns.cov() * 250
        cov_with_nifty = cov.iloc[0, 1]

        nifty_var = sec_returns['^NSEI'].var() * 250

        self.st_beta = cov_with_nifty / nifty_var
        return self.st_beta

    def caPM_predict(self, predict_no_yrs=1, risk_free_return=0.04, plot=False):
        tick = "{}.NS".format(self.ticker)
        stock_price = round(wb.DataReader(tick, data_source='yahoo')['Adj Close'][-1], 2)
        predict_percentage = risk_free_return + self.beta() * 0.05

        for y in range(predict_no_yrs):
            stock_price *= 1 + predict_percentage

        if plot:
            stock_price_data = wb.DataReader(tick, data_source='yahoo', start=self.start_input)['Adj Close']
            stock_price_data = stock_price_data.to_frame()
            p_date = datetime.now() + relativedelta(years=predict_no_yrs)
            stock_price_data = stock_price_data.reset_index()
            stock_price_data['Date'] = stock_price_data['Date'].apply(lambda x: x.date())
            predict_data = pd.DataFrame(columns=['Date', 'Adj Close'])
            predict_data.loc[0] = [stock_price_data.iloc[-1]['Date'], stock_price_data.iloc[-1]['Adj Close']]
            predict_data.loc[1] = [p_date.date(), stock_price]

            plt.figure(figsize=(15, 8))
            sns.set_style("darkgrid")
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.title("{}".format(tick), fontsize=18)
            plt.tight_layout()
            plt.plot_date(x=predict_data['Date'], y=predict_data['Adj Close'], linestyle='solid', marker=None)
            plt.plot_date(x=stock_price_data['Date'], y=stock_price_data['Adj Close'], linestyle='solid', marker=None)
            plt.legend(labels=["Expectation", 'Declared'])
            plt.show()

        return round(stock_price, 2)

    def plot_stock(self, nifty_50=False, num_years=5):
        tick = "{}.NS".format(self.ticker)
        if nifty_50:
            tickers = [tick, '^NSEI']
        else:
            tickers = [tick]

        plot_data = pd.DataFrame()
        start_date = datetime.now() - relativedelta(years=num_years)
        self.start_input = "{0}-{1}-{2}".format(start_date.year, start_date.month, start_date.day)

        for st in tickers:
            plot_data[st] = wb.DataReader(st, data_source='yahoo', start=self.start_input)['Adj Close']

        plot_data = plot_data.reset_index()
        print(plot_data)

        try:
            plot_data['index'] = plot_data['index'].apply(lambda x: x.date())

        except Exception:
            plot_data['index'] = plot_data['Date'].apply(lambda x: x.date())

        if not nifty_50:
            plt.figure(figsize=(15, 8))
            sns.set_style("darkgrid")
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Price', fontsize=14)
            plt.title("{}".format(tick), fontsize=18)
            plt.tight_layout()
            plt.plot_date(x=plot_data['index'], y=plot_data[tick], linestyle='solid', marker=None)
            plt.show()

        if nifty_50:
            log_plt = plot_data
            log_plt = log_plt.drop('index', axis=1)
            log_plot = (log_plt / log_plt.iloc[0] * 100)

            plt.figure(figsize=(15, 8))
            sns.set_style("darkgrid")
            plt.xlabel('Time', fontsize=14)
            plt.ylabel('Increase (%)', fontsize=14)
            plt.title("{} vs Nifty50 \n {} year Chart".format(tick, num_years), fontsize=18)
            plt.tight_layout()
            plt.xticks([])
            sns.lineplot(data=log_plot, dashes=False)
            plt.legend(labels=[tick, 'NIFTY50'])
            plt.show()


if __name__ == '__main__':
    t = FinancialData("TCS")
    # # t.make_predictions(t.net_cash_flow(as_list=True), num_terms_pred=2, plot=True)
    # t.discounted_cash_flow_price_predictor(predict_no_yrs=1, plot=True)
    t.plot_stock()
    #