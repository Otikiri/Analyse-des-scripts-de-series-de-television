import re 
import nltk
nltk.download('stopwords',quiet=True)
nltk.download('punkt_tab',quiet=True)
nltk.download('wordnet',quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

#===================================================================
# MOT_VIDE : ELEMENTS A ENLEVER DES TOKENS
#===================================================================

# Les mots vides doivent etre enleves pour avoir une analyse correcte 
motVide = set(stopwords.words('english')) 

# Mots vides additionnels
motVide.update([
    'dont', 'doesnt', 'didnt', 'cant', 'wont', 'wouldnt', 'couldnt', 'shouldnt',
    'yeah', 'okay', 'well', 'right', 'hey', 'know', 'yknow', 'youre', 'going',
    'really', 'like', 'get', 'gon', 'got', 'oh', 'uh', 'um', 'yes', 'no',
    'just', 'actually', 'wait', 'look', 'come', 'think', 'said', 'say', 'tell',
    'let', 'want', 'way', 'good', 'mean', 'little', 'thing', 'something','theres','thats','could'
    ,'would', 'back', 'time', 'great', 'guys', 'make','play','dude','theyre','shes','hes','whats'
])
motVide.update([
    'anyway', 'cause', 'ever', 'funny', 'getting', 'guess', 
    'hard', 'hell', 'hello', 'honey', 'isnt', 'kinda', 'leave',
    'might', 'morning', 'next', 'nice', 'night', 'nothing', 
    'pretty', 'ready', 'since', 'still', 'stop', 'stuff', 'sure',
    'together', 'totally', 'understand', 'whole', 'wrong', 'used',
    'wanted', 'went', 'tonight', 'thinking', 'talking',
    'umm', 'god', 'see', 'one', 'ive', 'guy', 'ill', 'got', 'know', 'like','wan','man',
    'na', 'nt', 'im', 'ur','man', 'wow', 'much', 'give',
    'always', 'another', 'anything', 'around', 'away',         
    'better', 'cause', 'even', 'every', 'fine', 'first',
    'sorry', 'thanks', 'thank', 'never', 'take', 'need',
    'listen', 'thought', 'best', 'place',"yeah","okay","ok","hey","hi","oh",
    "uh","um","well","thing","something","going","gonna","want","look","looking",
    "find","trying","start","keep",'huh'])
motVide.update([
    'people', 'talk', 'maybe', 'call', 'big', 'two', 
    'please', 'name', 'happened',
    'believe', 'day', 'girl', 'bad', 'whoa','alright','uhm'
])
motVide.update([
    'ross', 'rachel', 'monica', 'chandler', 'joey', 'phoebe',
])

# lemmatizer : permet de reduire des mots a leur racine
lemmatizer = WordNetLemmatizer()

# ==============================================================
# FONCTIONS POUR DATA LOADER 
# ==============================================================

# Prend en paramettre un script brute
# le nettoie : enleve la mise en scene, le prenom de l'interlocuteur, mets le texte en miniscule, 
# et enleve les espaces non necessaires
# Retourne le texte nettoyee.
def nettoyerTexte(texte): 
    texte = re.sub(r'\(.*?\)','',texte) # on envele la mise en scene
    texte = re.sub(r'^[A-Z\s]+:\s', '', texte,flags=re.IGNORECASE) # on enleve le nom de la personne qui parle
    texte = texte.lower() # on mets le texte en miniscule
    texte = re.sub(r'[^a-z\s]',' ', texte) #on remplace la ponctuation par des espaces
    texte = re.sub(r'\s+', ' ', texte).strip() # on enleve les espaces pas necesaires 
    texte = re.sub(r'\b(a+h+|o+h+|u+h+|m+h+)\b', '', texte)
    return texte 

# Prend en paramettre un script
# le tokenise et enleve les mots vide 
# renvoie un array de token
def tokeniserTexte(texte): 
    tokens= word_tokenize(texte)
    tokenNettoye = []
    for t in tokens: 
        if t not in motVide and len(t)>2 : 
            tokenNettoye.append(lemmatizer.lemmatize(t))
    return tokenNettoye

# ==============================================================
# FONCTIONS DIVERS
# ==============================================================

# Prend en parametre un dataframe
# split les phrases en mots 
# renvoie un array de mots nettoyer
def retournerTtMots(df):
    ttMots = []
    for texte in df['texte nettoyer']: 
        for word in texte.split(): 
            ttMots.append(word)
    return ttMots

# Prend en parametre un dataframe 
# renvoie un array de tokens
def retournerTokens(df):
    ttMots = []
    for texte in df['token']: 
        for word in texte: 
            ttMots.append(word)
    return ttMots

# (OBSOLETE : TO DELETE)
# Prend un array de tokens 
# renvoie une liste de tokens 
def joinTokens(tokens):
    return ' '.join([w for token in tokens for w in token])



def plotEvolutionMots(df):
    plt.figure(figsize=(10, 6))
    for mot in df['mot'].unique():
        data = df[df['mot']==mot]
        plt.plot(data['saison'],data['frequence_%'],label=mot,marker='o')
    plt.xlabel('Saison')
    plt.ylabel('Frequency %')
    plt.title('Evolution des frequences de mots au cours de la serie')
    plt.legend()
    plt.tight_layout()
    plt.savefig('freqMotsCoursTemps.png')
    plt.close()
