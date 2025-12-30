import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import nltk
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer