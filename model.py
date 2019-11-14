class Naive_Bayes:

    def __init__(self):
        self.categories={'hockey': {}, 'nba': {}, 'leagueoflegends': {}, 'soccer': {}, 'funny': {}, 'movies': {}, 'anime': {}, 
            'Overwatch': {}, 'trees': {}, 'GlobalOffensive': {}, 'nfl': {}, 'AskReddit': {}, 'gameofthrones': {}, 
            'conspiracy': {}, 'worldnews': {}, 'wow': {}, 'europe': {},  'canada': {}, 'Music': {}, 'baseball': {}}
        self.categories_total_counter={'hockey': 0, 'nba': 0, 'leagueoflegends': 0, 'soccer': 0, 'funny': 0, 'movies': 0, 'anime': 0, 
            'Overwatch': 0, 'trees': 0, 'GlobalOffensive': 0, 'nfl': 0, 'AskReddit': 0, 'gameofthrones': 0, 
            'conspiracy': 0, 'worldnews': 0, 'wow': 0, 'europe': 0,  'canada': 0, 'Music': 0, 'baseball': 0}
        self.target_counter={'hockey': 0, 'nba': 0, 'leagueoflegends': 0, 'soccer': 0, 'funny': 0, 'movies': 0, 'anime': 0, 
            'Overwatch': 0, 'trees': 0, 'GlobalOffensive': 0, 'nfl': 0, 'AskReddit': 0, 'gameofthrones': 0, 
            'conspiracy': 0, 'worldnews': 0, 'wow': 0, 'europe': 0,  'canada': 0, 'Music': 0, 'baseball': 0, 'total': 0}
    
   
    
    

    def preprocess1(self, sentence):
        import string
        import nltk
        from nltk.stem import PorterStemmer
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        words_to_remove = set(stopwords.words("english"))
        words_to_remove.add('I')
        ps=PorterStemmer()
        
        sentence=sentence.translate(str.maketrans('', '', string.punctuation))
        sentence=sentence.lower()
        words = word_tokenize(sentence)
        word_list=[]
    
        for word in words:
            if word not in words_to_remove:
                word_list.append(ps.stem(word))
        return word_list
    
       
    def fit(self, X, y):
        self.categories={'hockey': {}, 'nba': {}, 'leagueoflegends': {}, 'soccer': {}, 'funny': {}, 'movies': {}, 'anime': {}, 
            'Overwatch': {}, 'trees': {}, 'GlobalOffensive': {}, 'nfl': {}, 'AskReddit': {}, 'gameofthrones': {}, 
            'conspiracy': {}, 'worldnews': {}, 'wow': {}, 'europe': {},  'canada': {}, 'Music': {}, 'baseball': {}}
        self.categories_total_counter={'hockey': 0, 'nba': 0, 'leagueoflegends': 0, 'soccer': 0, 'funny': 0, 'movies': 0, 'anime': 0, 
            'Overwatch': 0, 'trees': 0, 'GlobalOffensive': 0, 'nfl': 0, 'AskReddit': 0, 'gameofthrones': 0, 
            'conspiracy': 0, 'worldnews': 0, 'wow': 0, 'europe': 0,  'canada': 0, 'Music': 0, 'baseball': 0}
        self.target_counter={'hockey': 0, 'nba': 0, 'leagueoflegends': 0, 'soccer': 0, 'funny': 0, 'movies': 0, 'anime': 0, 
            'Overwatch': 0, 'trees': 0, 'GlobalOffensive': 0, 'nfl': 0, 'AskReddit': 0, 'gameofthrones': 0, 
            'conspiracy': 0, 'worldnews': 0, 'wow': 0, 'europe': 0,  'canada': 0, 'Music': 0, 'baseball': 0, 'total': 0}
        
        def Naive_Counter(word_list, target_category):

            for word in word_list:
                if word in self.categories[target_category]:
                    self.categories[target_category][word]+=1
                else:
                    self.categories[target_category][word]=1
        
                self.categories_total_counter[target_category]+=1
                self.target_counter[target_category]+=1
                self.target_counter['total']+=1
       
      
        for comment, subreddit in zip(X, y):
            Naive_Counter(self.preprocess1(comment), subreddit)
            

    
    
    def keywithmaxval(self, d): 
        v=list(d.values())
        k=list(d.keys())
        return k[v.index(max(v))]        
            
            
    def predict(self, x):
        import numpy as np
        x=self.preprocess1(x)
        predictions={}
        for i in self.categories:
            predictions[i]=0
        
        for i in self.categories:
        
            theta = self.target_counter[i] / self.target_counter['total']
            theta1=np.zeros(len(x))
            theta0=np.zeros(len(x))
            sum_right=0
            
            for j,word in enumerate(x):
            
                numerator=0
                denominator=0
                if word in self.categories[i]:
                    theta1[j]= (self.categories[i][word]) / (self.categories_total_counter[i])
                else:
                
                    theta1[j]=0
                
                for k in self.categories:
                
                    if k!=i: 
                        if word in self.categories[k]:                    
                            numerator+= (self.categories[k][word]) 
                        denominator += (self.categories_total_counter[k]) 
                    
                if numerator!=0:    
                    theta0[j]= (numerator)/(denominator)
                
                else:
                    theta0[j]=0
                
            
                if theta0[j]==0 and theta1[j]==0:
                    add=0
                
                elif theta0[j]==0:
                    add=5
                elif theta1[j]==0:
                    add=-1
                else:
                
                    add= np.log( (theta1[j]) / (theta0[j]))
                sum_right+=add
        
            predictions[i] = np.log(theta/(1-theta)) + sum_right
    
        return self.keywithmaxval(predictions)
