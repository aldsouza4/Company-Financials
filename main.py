import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from statistics import mean


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
    else:
        return "Something Broke"


def clean_ser(val):
    return (np.around((val.reshape(-1)), 2)).tolist()


class FinancialData(object):

    def __init__(self, ticker):
        self.ticker = ticker
        self.df = pd.read_html("https://www.screener.in/company/{}/consolidated/".format(self.ticker))

    def financial_results(self, name, display=False):

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

        if as_list:
            return input_list + predictions[:-1]

        if asDataFrame:
            data = pd.DataFrame(data=[input_list + predictions[:-1]], columns=self.column_list, index=[self.pd_index])
            return data

        if only_prediction_list:
            return predictions[:-1]

    # Get functions

    # -----------QUARTER DATA-----------------------------------------------------

    def quarter_revenue(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "quarter_revenue"
        t_name = "quarter"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def quarter_operating_profits(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "quarter_operating_profits"
        t_name = "quarter"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def quarter_operating_margins(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "quarter_operating_margins"
        t_name = "quarter"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def quarter_profit_before_tax(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "quarter_profit_before_tax"
        t_name = "quarter"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def quarter_net_profit(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "quarter_net_profit"
        t_name = "quarter"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    # -----------ANNUAL DATA-----------------------------------------------------

    def annual_revenue(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "annual_revenue"
        t_name = "annual"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def annual_operating_profits(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "annual_operating_profits"
        t_name = "annual"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def annual_operating_margins(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "annual_operating_margins"
        t_name = "annual"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def annual_profit_before_tax(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "annual_profit_before_tax"
        t_name = "annual"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def annual_net_profit(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "annual_net_profit"
        t_name = "annual"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def annual_eps(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "annual_eps"
        t_name = "annual"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    # ------------Balance Sheet Items--------------------------------

    def share_capital(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "share_capital"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def reserves(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "reserves"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def borrowings(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "borrowings"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def other_liabilities(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "other_liabilities"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def total_liabilities(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "total_liabilities"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def fixed_assets(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "fixed_assets"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def capital_work(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "capital_work"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def other_assets(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "other_assets"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def total_assets(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "total_assets"
        t_name = "balance"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    # ------------Cash Flow Items ------------------------------------

    def cash_from_operating_activity(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "cash_flow_operating_activities"
        t_name = "cash_flow"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def cash_flow_investing_activities(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "cash_flow_investing_activities"
        t_name = "cash_flow"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def cash_from_financing_activity(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "cash_flow_financing_activities"
        t_name = "cash_flow"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def net_cash_flow(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "net_cash_flow"
        t_name = "cash_flow"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def roce(self, as_list=False, plot=False, as_DataFrame=True):
        self.r_name = "roce"
        t_name = "ROCE"
        return self.__get_results(table_name=t_name, name=self.r_name, as_list=as_list, plot_inner=plot,
                                  asDataFrame=as_DataFrame)

    def disp_data(self, data, as_list=False, plot=False, as_DataFrame=True, average=False):
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

    def net_income_margins(self, as_list=False, plot=False, as_DataFrame=True, average = False):
        net_income = self.annual_net_profit(as_list=True)
        revenue = self.annual_revenue(as_list=True)
        nim = [x / y for x, y in zip(net_income, revenue)]
        nim = [element * 100 for element in nim]
        self.percentage = True
        self.r_name = 'net_income_margins'
        return self.disp_data(data=nim, as_list=as_list, plot=plot, as_DataFrame=as_DataFrame, average=average)


if __name__ == '__main__':

    t = FinancialData("STLTECH")
    t.make_predictions(t.net_income_margins(as_list=True), plot=True )
