# Magic Models: Using Machine Learning to Afford Hobbies in Grad School

## Introduction

In this project, I am going to attempt to use machine learning to predict Magic: The Gathering card prices.

Magic: The Gathering is a trading card game that has been around for about 25 years.  Every year, new cards are released from Wizards of the Coast that players can buy, put in their decks, and play each other.  The competition ranges from casual to professional players flying halfway around the world to compete for thousands of dollars of prize money.  If you're interested in learning more, check out this awesome [video](https://www.youtube.com/watch?v=Plr81gaUIr0).  For me, it's mostly playing at my local game store with a big tournament sprinkled in here and there.  

One of the challenges is that Magic cards work like any other commodity, as the demand goes up or down given a certain supply, the price of cards can also fluctuate.  In cases like old powerful cards, there is a very small supply but a large demand causing them to sometimes be worth more than a used car.  Though I am not buying those kinds of cards, it still can be expensive and I am still in graduate school.  Thus, I wondered if I could use machine learning to help see when cards are going to spike or fall in price.  I know, I know you're thinking here's another person trying to predict something like the stock market.  I would, however, argue that the Magic economy is a bit more contained than the stock market in that there are less variables affecting prices.  Regardless of the outcome, this is still a great opportunity and use case to study time-series data, web scraping, and yes even neural networks.

For clarity, I have broken down the project into different sections.  Feel free to read them in or out of order depending on your interest.

## Table of Contents

[Magical_Data_Engineering_P1](https://github.com/desdelgado/Magic_Models/blob/master/Magical_Data_Engineering_P1.ipynb) - A first pass at web scraping a single card and transforming the data so it's usable for modeling.

[Single_Timestep_Univariable_ Model_P2](https://github.com/desdelgado/Magic_Models/blob/maste/Single_Timestep_Univariable_%20Model_P2.ipynb) - Predicting the card price one day ahead. 
