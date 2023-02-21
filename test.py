import spacy

nlp = spacy.load("pl_nask")
txt = """Do 1923 roku był otrzymywany wyłącznie przez destylację rozkładową drewna. Obecnie produkuje się go syntetycznie, głównie dwiema metodami z gazu syntezowego:

    metodą ICI
    metodą Lurgi.

Główną reakcję tych procesów przedstawia równanie:

    CO + 2H2 → CH3OH

Reakcja ta prowadzona jest w obecności katalizatora miedziowego (Cu-Zn-Al2O3), w temperaturze 250 °C, przy ciśnieniu 4–10 MPa. Wcześniej stosowane katalizatory chromowo-cynkowe wymagały 340–400 °C oraz 30–32 MPa.

W trakcie procesu przebiega równocześnie reakcja wodoru z dwutlenkiem węgla (potrzebnym do utrzymania aktywności katalizatora):

    CO2 + 3H2 → CH3OH + H2O

Alkohol metylowy można także otrzymać, działając NaOH lub KOH na fluorowcopochodne metanu, na przykład CH3Cl, CH3Br, CH3I, CH3F:

    CH3Cl + KOH → CH3OH + KCl

Metanol jest używany w zakładach przemysłowych jako rozpuszczalnik i surowiec do otrzymywania aldehydu mrówkowego, chlorku metylu, barwników.

W mroźny wieczór, ostatni dzień mijającego roku, mała dziewczynka wędruje boso ulicami miasta, nadaremno próbując sprzedać zapałki. Żaden z przechodniów nie jest zainteresowany zmarzniętym i głodnym dzieckiem, każdy w natłoku ostatnich przygotowań do świętowania z rodzinami w pośpiechu udaje się do domu. Zrezygnowana i zniechęcona dalszą pracą, znajduje cichy kąt pomiędzy dwoma domami, gdzie siada, aby rozgrzać zziębnięte nogi. Rozmarzona myślą poczucia ciepła, rozpala kolejno po sobie cztery zapałki. Każda z nich jest symbolem jej pragnień, które przez kilka sekund widzi własnymi oczami. Jako pierwszy ukazuje się żelazny piec, przy którym mogłaby ogrzać zziębnięte skostniałe dłonie, następnie pieczona gęś, którą najadłaby się do syta, jak nigdy wcześniej w swoim życiu. Kolejno pojawia się choinka, oświetlona świeczkami, które ją ogrzewały. Zapałka zgasła, a kilka iskierek uniosło się w górę, ku niebu. Jedna z nich zaczęła spadać niczym gwiazda. Dziewczynka przypomniała sobie, jak jej babcia opowiadała, że to znak zapowiadający nadejście śmierci. Dziewczynka zapala następną zapałkę i pojawia się jej ukochana babcia. Bojąc się, że zniknie, od razu zapala całą paczkę zapałek. Razem unoszą się do miejsca, gdzie nie ma głodu i zimna, przed tron Boga[1][2].

Spotkałam wówczas go m.in. w piekarni. Kupował bułki. Powiedziałabym wtedy: "Widzisz mnie?".

C. Ciamciara, op. cit., s. 97.

Dz.U. 2023 poz. 6 Rozporządzenie Ministra Klimatu i Środowiska z dnia 30 grudnia 2022 r. zmieniające rozporządzenie w sprawie wzoru oświadczenia odbiorcy uprawnionego.

"""

doc = nlp(txt)

for tok in doc:
    print(tok.i, tok.orth_, tok.lemma_, tok.head, tok.dep_, tok._.diminutive_chain, tok.tag_)
