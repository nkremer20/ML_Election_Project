#importing libraries
import pandas as pd

#import excel files
election1996 = pd.read_excel(r'G:\Python\Project\Data\Iowa\1996\Iowa_1996_Votes.xlsx')
election2000 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2000\Iowa_2000_Data.xlsx')
election2004 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2004\Iowa_2004_Data.xlsx')
election2008 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2008\Iowa_2008_Data.xlsx')
election2012 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2012\Iowa_2012_Data.xlsx')
election2016 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2016\Iowa_2016_Data.xlsx')
election2020 = pd.read_excel(r'G:\Python\Project\Data\Iowa\2020\Iowa_2020_Data.xlsx')

#Calculate the winning party of the county
election2020.winner = 0
election2020.winner[election2020.total_rep > election2020.total_dem] = 1

election2016.winner = 0
election2016.winner[election2016.total_rep > election2016.total_dem] = 1

election2012.winner = 0
election2012.winner[election2012.total_rep > election2012.total_dem] = 1

election2008.winner = 0
election2008.winner[election2008.total_rep > election2008.total_dem] = 1

election2004.winner = 0
election2004.winner[election2012.total_rep > election2012.total_dem] = 1

election2000.winner = 0
election2000.winner[election2000.total_rep > election2000.total_dem] = 1

election1996.winner = 0
election1996.winner[election1996.total_rep > election1996.total_dem] = 1

#Winner of previous election
election2016.prev_winner = election2012.winner

election2012.prev_winner = election2008.winner

election2008.prev_winner = election2004.winner

election2004.prev_winner = election2000.winner

election2000.prev_winner = election1996.winner

#Writing all calculations to the spreadsheets
final_2020 = election2020.to_excel(r'G:\Python\Project\Data\Iowa\2020\Iowa_2020_Data.xlsx')

final_2016 = election2016.to_excel(r'G:\Python\Project\Data\Iowa\2016\Iowa_2016_Data.xlsx')

final_2012 = election2012.to_excel(r'G:\Python\Project\Data\Iowa\2012\Iowa_2012_Data.xlsx')

final_2008 = election2008.to_excel(r'G:\Python\Project\Data\Iowa\2008\Iowa_2008_Data.xlsx')

final_2004 = election2004.to_excel(r'G:\Python\Project\Data\Iowa\2004\Iowa_2004_Data.xlsx')

final_2000 = election2000.to_excel(r'G:\Python\Project\Data\Iowa\2000\Iowa_2000_Data.xlsx')