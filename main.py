import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


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


class FinancialData(object):

    def __init__(self):
        self.df = pd.read_html("https://www.screener.in/company/HDFC/consolidated/")

    def financial_results(self, name, display=False):

        data = None

        if name == "quarter":
            table_index = 0
            data = pd.DataFrame(self.df[table_index])

        if name == "annual":
            table_index = 1
            data = pd.DataFrame(self.df[table_index])
            data.drop("TTM", axis=True, inplace=True)

        else:
            print("Something Broke")


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

        column_list = data.columns
        column_list = column_list[1:]
        column_list = column_list.tolist()

        if name == "quarter_revenue":
            row_index = 0
        elif name == "quarter_operating_profits":
            row_index = 2
        elif name == "quarter_operating_margins":
            row_index = 3
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

            sns.lineplot(x=index_list, y=data)
            plt.show()

        if as_list:
            return data

        if asDataFrame:
            if name == "quarter_revenue":
                pd_index = "Quarter Revenue"
            elif name == "quarter_operating_profits":
                pd_index = "Quarter Operating Profits"
            elif name == "quarter_operating_margins":
                pd_index = "Quarter Operating Margins(%)"
            elif name == "quarter_profit_before_tax":
                pd_index = "Quarter Profit Before Tax"
            elif name == "quarter_net_profit":
                pd_index = "Quarter Net Profit"

            elif name == "annual_revenue":
                pd_index = "Annual Revenue"
            elif name == "annual_operating_profits":
                pd_index = "Annual Operating Profits"
            elif name == "annual_operating_margins":
                pd_index = "Annual Operating Margins(%)"
            elif name == "annual_profit_before_tax":
                pd_index = "Annual Profit Before Tax"
            elif name == "annual_net_profit":
                pd_index = "Annual Net Profit"
            elif name == "annual_eps":
                pd_index = "Annual EPS"
            else:
                return "Something Broke"

            data = pd.DataFrame(data=[data], columns=column_list, index=[pd_index])
            return data

    # -----------ANNUAL DATA-----------------------------------------------------

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
print(t.annual_net_profit(plot=True))
