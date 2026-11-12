# PW Podstawy Przetwarzanie Danych
## Kategoryzacja wiadomości AG News

### Notatki

- Chcemy na którymś etapie przeprowadzić analizę czy modele generalizują się na nowe artykuły (z innego okresu czasu)
- Jest kwestia czy jakoś kwantyfikujemy i uwzględniamy pewność modeli w klasyfikacji

### Do zrobienia

- [x] Dataset
- [x] EDA (Jest stara i nowa - czy coś jeszcze robimy z nową?)
- [ ] Model B0
- [ ] Model B1 (In Progress - Jarek)
- [x] Przetwarzanie P1, P2, P3
- [ ] Przygotowanie 5 slajdów z wynikami na Milestone 2
 
**⬆️ Do zrobienia na 12.11.26 ⬆️**

- [ ] Eksperymenty nad przetwarzaniem P1, P2, P3,
- [ ] Model M
- [ ] Krótka Prezentacja na Milestone 3

**⬆️ Do zrobienia na 03.12.26 ⬆️**

- [ ] Model L
- [ ] Eksperymenty, tabele i wyniki do raportu - wstępnie
- [ ] Krótka Prezentacja na Milestone 4

**⬆️ Do zrobienia na 14.01.27 ⬆️**

- [ ] Raport
- [ ] Prezentacja

**⬆️ Do zrobienia na 28.01.27? ⬆️**

### Treść Zadania

Zadanie: Klasyfikacja wiadomości: World, Sports, Business, Sci/Tech.
Zakres: Z oficjalnego treningu wybrać ustaloną próbkę do 20 tys. rekordów i wydzielić walidację; zachować oficjalny test.

Model bazowy: TF-IDF + regresja logistyczna lub LinearSVC.
Przykładowe eksperymenty: Normalizacja HTML i znaków; normalizacja liczb; usuwanie lub zachowanie stop-słów. Osobny eksperyment: tytuł, opis i ich połączenie.

Metryki: Macro-F1 (główna), accuracy i macierz pomyłek.
Uwaga: Porównanie pola tytułu i opisu wymaga wersji zachowującej te pola. Zapisać źródło konkretnej paczki i warunki użycia danych; licencji kodu repozytorium nie utożsamiać z licencją artykułów.

# Milestony

| Spotkanie | Prezentowany rezultat | Rodzaj |
| --- | --- | --- |
| **1. Początkowe** | Organizacja, skład zespołów, przydział tematów, kryteria oceniania i harmonogram. | Organizacyjne |
| **2. Milestone 1** | Dane i EDA: etykiety, liczebności, braki, duplikaty, podział; trzy hipotezy dotyczące przetwarzania. | Merytoryczne |
| **3. Milestone 2** | Działające B0 i B1; pierwsze wyniki P1-P3, konfiguracje i plan dalszych porównań. | Merytoryczne |
| **4. Milestone 3** | Pełne porównanie wariantów przetwarzania oraz wstępne wyniki modelu M. Ocena jakości i metodologii, wskazanie braków. | Oceniające - etapowe |
| **5. Milestone 4** | Porównanie z LLM, analiza niepewności i błędów, koszty; robocza tabela wyników i wnioski do raportu. | Merytoryczne |
| **6. Końcowe** | Prezentacja całego projektu, końcowe wyniki, wnioski, ograniczenia i indywidualne odpowiedzi na pytania. | Oceniające - końcowe |

### Cel Biznesowy

Klasyfikator wiadomości na platformie typu agregator wiadomości (dobór reklam, sugerowanie innych artykułow, tworzenie zbiorczych zestawień tematycznych)

### Preztwarzanie i Modele

- B0
  Klasyfikator przypisujący każdemu wejściu najczęstszą klasę zbioru treningowego przed zbalansowaniem (Sci/Tech; `train.csv` jest zbalansowany, więc tam wszystkie klasy są równie częste).
- B1
  TF-IDF + LinearSVC
- P1
  Usunięcie artefaktów HTML
- P2
  Usunięcie StopWordów
- P3
  Usunięcie wszystkich słów nienależących do standardowego słownika (a więc też nazw własnych itd.)
- P4
  Usunięcie słów bardzo częstych/bardzo rzadkich
- P5
  Warianty: tytuł+opis, tytuł, opis
- M
  
  Może być:

  - Embedding SentenceTransformer lub podobne i prosty model klasyfikujacy
  - BERT wytrenowany do naszych kategorii
  
  A może oba
- L
  LLM w wariantach zero-shot i few-shot

# Dane i odtworzenie wyników

### Dane

- Źródło: oryginalny korpus AG's corpus of news articles (z niego zbudowano benchmark AG News): <http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html>
- Pobranie: <http://groups.di.unipi.it/~gulli/newsSpace.bz2> (plik `newsSpace.bz2` zapisać w katalogu głównym repozytorium; nie jest w gicie)
- Warunki użycia: według opisu korpusu (cytowanego też w `eda/old_ag_news_csv/readme.txt`) dane udostępniono do celów badawczych i innej działalności niekomercyjnej. Licencja kodu repozytorium nie obejmuje artykułów.

### Odtworzenie wyników

Wszystkie polecenia uruchamiać z katalogu głównego repozytorium. Zależności: `uv sync` (Python 3.13, `pyproject.toml`), potem polecenia przez `uv run` albo w aktywnym środowisku.

```bash
# 1. Dane: raw_newsspace.csv i newsspace_era_b.csv (do EDA) oraz splits/ (train, val, test, test-future)
python prepare_newsspace.py newsSpace.bz2 --raw --full --train

# 2. B0, B1 i warianty P; wyniki w results/<czas>-<id>/ (P2 przy pierwszym uruchomieniu pobiera stop-słowa NLTK)
python runner.py --models models/b0.json models/b1.json models/p1b1.json models/p2b1.json models/p3b1.json models/p4b1.json

# 3. Testy
python -m unittest discover -s tests
```

Ocenę na zbiorze testowym (`--evaluate-test`) uruchamiać dopiero po zamrożeniu wyboru wariantów.

Notatniki w `eda/` korzystają z plików z kroku 1 i działają uruchomione zarówno z `eda/`, jak i z katalogu głównego: `newsspace_analysis.ipynb` (cały korpus, ery A i B), `eda_era_b.ipynb` (tekst klas modelowanych w erze B), `old_eda.ipynb` (archiwalne EDA na 4-klasowym AG News).
