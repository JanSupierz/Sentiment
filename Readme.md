# IMDb Review Sentiment Analysis

This project focuses on sentiment analysis of IMDb movie reviews.
The primary emphasis is on text normalization, n-gram feature engineering, and filtering strategies prior to TF-IDF–based modeling and BERT usage.

---

## Installation

```
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset: IMDb Reviews

IMDb reviews frequently contain HTML markup and non-standard typography.

Typical issues include:

* HTML tags such as `<br />`
* Multiple representations of the same symbol:

  * Quotes: `´`, `’`, `‘` → `'`
  * Ellipsis: `…` → `...`
  * Dashes: various Unicode dashes → `-`

All punctuation is intentionally preserved because it carries sentiment information.
For example, sequences like `..` and `?` are strong indicators of negative sentiment according to empirical results.

---

## Raw Review Example

> I first watched Kindred in 1987 along with another movie called devouring waves. I remember back then i hated them both and i have never really bothered to watch them again.<br /><br /> However i have recently started a crusade to collect as many 80's horror titles in their original boxed form, That have been deleted for some time. I have got myself quite a proud collection with many more titles on my list!<br /><br />The Kindred although i have not as yet got a copy is high priority as all the old movies i didn't like back then, I now own and have now re-watched and think they are brilliant and the bits i do remember of the Kindred are now driving me to want to get hold of a copy A.S.A.P.<br /><br />Hurray for the 80's and long live horror!

---

## Text Cleaning

After HTML removal and normalization of non-standard characters, the review becomes:

> I first watched Kindred in 1987 along with another movie called devouring waves. I remember back then i hated them both and i have never really bothered to watch them again. However i have recently started a crusade to collect as many 80's horror titles in their original boxed form, That have been deleted for some time. I have got myself quite a proud collection with many more titles on my list! The Kindred although i have not as yet got a copy is high priority as all the old movies i didn't like back then, I now own and have now re-watched and think they are brilliant and the bits i do remember of the Kindred are now driving me to want to get hold of a copy A.S.A.P. Hurray for the 80's and long live horror!

This cleaned version is used for BERT-based models.

---

## Normalization and Contraction Removal

The next step includes:

* Lowercasing all words
* Expanding contractions

  * `"didn't"` → `"did not"`

Result:

> i first watched kindred in 1987 along with another movie called devouring waves. i remember back then i hated them both and i have never really bothered to watch them again. however i have recently started a crusade to collect as many 80's horror titles in their original boxed form, that have been deleted for some time. i have got myself quite a proud collection with many more titles on my list! the kindred although i have not as yet got a copy is high priority as all the old movies i did not like back then, i now own and have now re-watched and think they are brilliant and the bits i do remember of the kindred are now driving me to want to get hold of a copy a.s.a.p. hurray for the 80's and long live horror!

---

## N-Gram Generation

N-grams are sequences of consecutive words.

Example sentence:

```
it is a fun movie
```

Generated n-grams:

* Unigrams
  it, is, a, fun, movie, .

* Bigrams
  it is, is a, a fun, fun movie

* Trigrams
  it is a, is a fun, a fun movie, fun movie .

Many of these n-grams are not informative, which motivates filtering.

---

## N-Gram Filtering

### Stop-Units

* The n-grams that appear in more than **30% of documents per label** are identified
* If an n-gram appears frequently in **both** positive and negative labels, it is removed
* These tokens do not carry sentiment information

Examples of removed tokens:

* this
* that
* there
* is
* not
* good

More informative examples:

* not good
* very good

---

### Rare Tokens

* N-grams appearing **fewer than 10 times** in the entire corpus are removed
* These typically include:

  * Misspellings
  * Foreign words
  * Non-existent tokens

---

## Feature Engineering

Before building the TF-IDF pipeline:

1. Stop-units are removed
2. Rare tokens are removed
3. Correlation between n-grams and sentiment polarity is computed

This process isolates sentiment-relevant features.

---

## Example Reviews and Label Issues

### Positive Reviews Labeled as Negative

> this is a great movie. I love the series on tv and so I loved the movie. One of the best things in the movie is that Helga finally admits her deepest darkest secret to Arnold!!! that was great. i loved it it was pretty funny too. It's a great movie! Doy!!!

> I enjoyed the beautiful scenery in this movie the first time I saw it when I was 9 . Dunderklumpen is kind of cute for kiddies in a corny way. It reminded me of HRPUFFINSTUFF on sat mornings, Its Swedish backdrops make it easy on the eyes . Don't expect older kids to be interested as the live action/animation is way behind the times and most older kids will get bored.This is definitely an under 10 age set movie and a nice bit of memories for those of us who were little kids in 1974.

*These reviews are clearly positive, but were labeled as negative.*

---

### Ambiguous Review (≈ 5/10)

> I am a back Batman movie and TV fan. I loved the show (new and old) and I loved all the movies. But this movie is not as great as some people were hopeing it to be. In my opinon, it is a big let down. I think the problem was it had no drama. Batman: Mask Of The Phantasm and Batman Beyond: Return Of The Joker had a lot of drama. and Batman & Mr. Freeze: Sub Zero had some drama too. Also, I think this movie is to light for Batman. The only scene that seems a little dark is the big fight with Bane at the end. Anyways, it's an ok Batman movie. But I would just rent it.

*This review is difficult to classify and close to neutral.*

---

### Mixed Review (Predicted as Negative)

> **SPOILERS AHEAD** It is really unfortunate that a movie so well produced turns out to be such a disappointment. I thought this was full of (silly) clichés and that it basically tried to hard. To the (American) guys out there: how many of you spend your time jumping on your girlfriend's bed and making monkey sounds? To the (married) girls: how many of you have suddenly gone from prudes to nymphos overnight--but not with your husband? To the French: would you really ask about someone being "à la fac" when you know they don't speak French? Wouldn't you use a more common word like "université"? I lived in France for a while and I sort of do know and understand Europe (and I love it), but my (German) roommate and I found this pretty insulting overall. It looked like a movie funded by the European Parliament, and it tried too hard basically. It had all sorts of differences that it tried to tie together (not a bad thing in itself) but the result is at best awkward, but in fact ridiculous--too many clashes that wouldn't really happen. Then the end of the movie--the last 10 minutes--ruined all the rest. Why doesn't Xavier talk to the Erasmus students he meets back in Paris? Why does he just walk off? Why does he just run away from his job, is that "freedom"? And in the end, is the new Europe supposed to rest on a bunch of people who smoke up and shag all day? Is this what it's made up of? Besides, the acting was pretty horrible. I can't believe Judith Godrèche's role and acting. Why was she made to look like Emanuelle Béart so much? At first I thought Xavier was OK but with retrospect I think he was pretty bad. And that's all really too bad, because technically (opening credits, scenes when he's asking what papers he needs) it was really good (except for sound editing around the British siblings), and the soundtrack was great too. So the form was good, but the content pretty horrible.

*The sentiment is mixed, but overall negative.*

---

### Plot-Heavy Reviews

> Hollywood has turned the Mafia in to a production line of output ranging from the banal to the excellent and despite some good acting and a reasonable script (much of which is - for a change - true!) this "home entertainment" effort has to fall slap bang in the middle. The script is not only obvious (all of the checklist boxes end up being ticked), but spends a lot of time trying to create a pastiche of the best of other people's work. The Godfather being the most obvious, but there are other references too. I won't bother naming them. Nevertheless it is a good taste borrower! The producer seems to set a quota for gunshots and murder (one at least every twenty minutes?) and the ending is weak and "so what?" I am told there are various versions of this production so that maybe that is just the version I have seen. Gangsters don't make money they take money. Usually by fear. Some seem more in to the murder and mayhem side of the business than making money. They were the ones that were the first to go (in real life and here). "You can't make money with a gun in your hand" says Charlie 'Lucky' Luciano at one stage. One of the smarter gangsters, although all things are relative. He was a skilled white slave trader and a drug dealer before being bundled home to Italy. The old school "moustached Pete's" were picked off by the new bloods who wanted the power and the money for themselves and to break free of the straight jacket of Italian/Sicilian power (rarely doing business outside themselves). The young Turk knew they needed to be allied with other groups (most notably "the Jews" who knew how to launder money) and this is at least referenced and acknowledged. What isn't made so clear is that most immigrant groups had their own Mafia's - but most of them made their money and went legit. And why not? Who wants to die in jail? Joseph Bonanno was a ruthless man prepared to kill if needs be , but not an unfair or stupid one. His story was tragic in that he could have made money in the over ground world and he showed a special skill in avoiding getting killed. With a little bit of luck attached, naturally. Despite the range of respectable names and three actors in the title role (Bruce Ramsay, Martin Landau and Tony Nardi) there isn't the charisma or the talent to bring us in and feel anything. We are - merely - passive observers in a life we are glad not to have lead. The people shown here were born in to a cruel world but their only mark was to make it crueler. If you can't get enough of the gangster genre that will be better than watching Godfather 1 & 2 for the tenth time and it is even better -- as basic entertainment -- than the horrible misfire that was Godfather 3.

*Plot explanation and comparisons to other movies are among the hardest cases.*

---

### Potential Sarcasm

> It's a good movie if you plan to watch lots of landscapes and animals, like an animal documentary. And making Pierce Brosnan an indian make you wonder 'Does all those people don't recognize if someone isn't indian at plain sight?'

*Some reviews are sarcastic, others literal — a known challenge in sentiment analysis.*

---
