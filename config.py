EMAIL = "email@email.com" # for crossref api requests

# H-INDEX, SJR2024
ANTHROPOLOGY_JOURNAL_LIST = ["American_Anthropologist", # 105, 0.802 Q1
                            "American_Ethnologist", # 101, 1.197 Q1
                            "American_Journal_of_Anthropology", # 
                            "Annual_Review_of_Anthropology", # 151, 0.978 Q1
                            "Anthropological_Quarterly", # 59, 0.242 Q2
                            "Anthropological_Theory", #67, 0.838 Q1
                            "Anthropology_Today", # 36, 0.756 Q1
                            "Cambridge_Journal_of_Anthropology", # 7, 0.247 Q2
                            "Cultural_Anthropology", # 96, 0.983 Q1
                            "Ethnography", # 61, 0.588 Q1
                            "Ethnos", # 56, 0.789 Q1
                            "Ethos", # 56, 0.323 Q2
                            "Journal_of_Contemporary_Ethnography", # 69, 0.504 Q1
                            "Journal_of_the_Royal_Anthropological_Institute", # 80, 0.854 Q1
                            "Medical_Anthropology_Quarterly", # 69, 0.770 Q1
                            "Social_Anthropology_Anthropologie_Socaile" # 54, 0.326 Q2
                            ]

SOCIOLOGY_JOURNAL_LIST = [ 
                         "Demography", # 158 Q1, 2.363
                         "Socius", # 43 Q1, 1.185
                         "Sociology", # 138 Q1, 1.497
                         "Gender & Society" # 132 Q1, 2.272
                         "Social Forces", # 158 Q1, 1.566
                         "American Journal of Sociology" # 214 Q1, 2.811
                         "Sociological Theory" # 94 Q1, 2.527
                         "Theory & Society" # 94 Q1, 1.047
                         ]

PSYCHOLOGY_JOURNAL_LIST = [
                          "Nature_Human_Behaviour", # 113 Q1, 5.537
                          "Journal_of_Personality_and_Social_Psychology", # 447, 3.865
                          "Trends_in_Cognitive_Sciences", # 375, 4.506
                          "Frontiers_in_Psychology", # 212 Q2, 0.872
                          "Psychological_Science", # 331, 2.500
                          "Journal_of_Experimental_Psychology_General", # 196, 2.076
                          "Journal_of_Applied_Psychology", # 353, 6.803
                          ]

EVOANTHRO_JOURNAL_LIST = [
                          "Evolutionary_Anthropology", # 101, 1.405
                          "Journal_of_Human_Evolution", # 145, 1.393
                          "Evolutionary_Human_Sciences", # 18, 0.991
                          "Archaeological_and_Anthropological_Sciences", # 47, 0.888
                          "American_Journal_of_Biological_Anthropology", # 145, 0.861
                          "American_Journal_of_Human_Biology", # 97, 0.676
                          "Journal_of_the_Royal_Anthropological_Institute", # 80, 0.854
                          ]


SELECTED_ANTHROPOLOGY_JOURNALS = ["American_Anthropologist", "American_Ethnologist", "Cultural_Anthropology", "Ethnography", "Journal_of_Contemporary_Ethnography", "Journal_of_the_Royal_Anthropological_Institute"]
SELECTED_SOCIOLOGY_JOURNALS = ["Demography", "Sociology", "American_Journal_of_Sociology", "Social_Forces", "Gender_and_Society", "Sociological_Theory"]
SELECTED_PSYCHOLOGY_JOURNALS = ["Nature_Human_Behaviour", "Journal_of_Personality_and_Social_Psychology", "Trends_in_Cognitive_Sciences", "Psychological_Science", "Journal_of_Experimental_Psychology_General", "Journal_of_Applied_Psychology"]
SELECTED_EVOANTH_JOURNALS = ["American_Journal_of_Biological_Anthropology", "American_Journal_of_Human_Biology", "Archaeological_and_Anthropological_Sciences", "Evolutionary_Anthropology", "Evolutionary_Human_Sciences", "Journal_of_Human_Evolution"]

SELECTED_JOURNALS = SELECTED_EVOANTH_JOURNALS
FIELD_NAME = "evoanth"
MAX_TEAM_SIZE = 100
MIN_YEAR = 1925
MAX_YEAR = 2024
ROLLING_WINDOW = 5