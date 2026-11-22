# Fachgespräch: Fragen, die kommen werden

**So benutzt ihr diese Liste:** Erst laut und in eigenen Worten antworten, dann hier vergleichen.
Die Kernpunkte sind Stichworte, kein Text zum Auswendiglernen. Im Fachgespräch zählt, dass ihr es
erklären und im Code zeigen könnt. Am besten fragt ihr euch gegenseitig ab.

Die Abschnittsnummern beziehen sich auf die Projektpipeline des Kurses.

---

## A. Forschungsfrage und Einordnung

**1. Was ist eure Forschungsfrage?**
- Rund 25 bekannte Personen (geschlossene Menge). Eingabe: ein Gehen-Clip mit vier Sensoren. Ausgabe: die Personen-ID.
- Zweiter Teil: Wie viel davon ist wirklich die Person, wie viel nur dieselbe Aufnahme oder derselbe Sensorsitz?

**2. Warum ist das kein Sicherheitssystem?**
- Das Modell kennt nur Personen aus dem Training und kann nicht „unbekannt“ sagen.
- Wenige Personen, ein Messtag, kleine Datenmenge. Die Pipeline nennt es ausdrücklich explorativ (Abschnitt 1).

**3. Was ist die größte Schwäche eurer Daten?**
- Pro Person und Aktivität eine Aufnahme, alle am selben Tag. Training und Test kommen aus derselben Aufnahme.
- Das Modell könnte deshalb Sensorsitz, Gerät oder Tagesform lernen statt des Gangs (Session-Risiko, Abschnitt 7).
- Was wir dagegen tun: Cross-Activity-Test und Lage-Test. Was eigentlich nötig wäre: ein zweiter Messtag.

## B. Daten und Vorverarbeitung (`laden.py`)

**4. Wie kommt ihr von den Dateien zu einem Trial?**
- Der Dateiname folgt der Namenskonvention des Kurs-Repos. Ein regulärer Ausdruck (`DATEINAME`) zerlegt ihn in Person, Aktivität, Trial, Gerät und Position (`finde_dateien`).
- Pro Trial werden die vier Sensordateien gelesen (`lies_datei`) und synchronisiert (`synchronisiere`).

**5. Wie synchronisiert ihr die Sensoren?**
- Gemeinsamer Zeitbereich: vom spätesten Start bis zum frühesten Ende.
- Darauf ein gemeinsames 50-Hz-Raster, Werte per linearer Interpolation (`np.interp`).
- Das geht nur mit absoluten Zeitstempeln; deshalb bevorzugt `_finde_zeitspalte` epoch- oder timestamp-Spalten.

**6. Warum ein Tiefpass vor dem Heruntertakten?**
- Nyquist: Mit 50 Hz lassen sich nur Frequenzen bis 25 Hz darstellen. Schnellere Anteile würden als falsche, langsame Schwingungen auftauchen (Aliasing).
- Butterworth 4. Ordnung bei 20 Hz. `filtfilt` filtert vorwärts und rückwärts, dadurch wird das Signal nicht zeitlich verschoben (`_tiefpass`).

**7. Was passiert mit Datenlücken?**
- Lücken über 0,1 s werden nicht überbrückt, die Werte dort werden NaN (Abschnitt 4: Lücken nicht unbemerkt interpolieren).
- Fenster, die NaN enthalten, werden verworfen (`schneide_fenster`). Der Qualitätsbericht zeigt Anzahl und Länge der Lücken.

**8. Welche Qualitätsprüfungen macht ihr?**
- Pro Datei: Dauer, Abtastrate, Lücken, Median des Beschleunigungsbetrags (sollte etwa 1 g sein), 99. Perzentil des Gyroskops.
- Daten in m/s² erkennt der Code am Betrag von etwa 9,81 und rechnet sie in g um.

**9. Warum schneidet ihr am Anfang und Ende 1 s ab?**
- Dort liegen oft Knopfdruck, Stillstehen oder Übergänge. Unklare Grenzen sollen laut Pipeline ausgeschlossen werden (Abschnitt 4).

## C. Fenster und Split (`fenster.py`): der wichtigste Teil

**10. Warum Fenster, warum 3 s mit 50 % Überlappung?**
- Die Modelle brauchen Eingaben fester Länge.
- 3 s enthalten mehrere Schritte (ein Schritt dauert etwa eine halbe Sekunde) und liegen im Startbereich der Pipeline (2 bis 3 s, Abschnitt 5).
- Überlappung ergibt mehr Trainingsbeispiele. Sie ist erlaubt, solange überlappende Fenster nicht auf Training und Test verteilt werden.

**11. Was sind Clips, und warum 5?**
- Jeder Trial wird in 5 gleich lange Clips geteilt. Der Clip ist die Einheit, über die entschieden wird („wer geht in diesem Clip?“), und die Einheit, die im Test landet.
- Die Clip-Grenzen werden vor den Fenstern gezogen; kein Fenster reicht über eine Grenze (Abschnitt 8).
- 5 ist ein Kompromiss: Mehr Clips geben mehr Folds, aber kürzere Clips mit weniger Fenstern, und der Gap kostet anteilig mehr Daten.

**12. Was ist Data Leakage?**
- Information aus den Testdaten gelangt ins Training. Das Ergebnis sieht dann besser aus, als es ist.
- Bei überlappenden Fenstern: Verteilt man sie zufällig, liegen fast identische Fenster in Training und Test. Das Modell erkennt dann Daten wieder, statt zu verallgemeinern.

**13. Wie funktioniert euer Block-Split?**
- In Fold k ist Clip k jedes Trials der Test, die übrigen Clips sind Training. 5 Folds, jeder Clip ist genau einmal Test (`block_split`, `block_folds`).
- Alle Personen sind in Training und Test vertreten, wie es Abschnitt 7 verlangt.

**14. Wozu zusätzlich der Gap, und warum 3 s?**
- Auch ohne Überlappung ähneln sich zeitlich benachbarte Daten (gleicher Rhythmus, gleiche Kraft). Trainingsfenster, die näher als 3 s am Test-Clip liegen, fliegen deshalb raus.
- 3 s ist eine Fensterlänge, der Startwert aus Abschnitt 7.

**15. Wie prüft ihr, dass kein Leakage passiert?**
- `kleinster_abstand` nimmt alle Paare aus Trainings- und Testfenster im selben Trial und berechnet den zeitlichen Abstand: max(Beginn Test − Ende Training, Beginn Training − Ende Test). Negativ heißt Überlappung.
- `experimente.py` bricht ab, wenn der Abstand unter 3 s liegt (`pruefe_split`). Derselbe Check läuft als automatischer Test bei jedem Push.

**16. Was zeigt E2?**
- Dasselbe Modell mit verschiedenen Splits: zufällige Fenster (falsch), Block ohne Gap, Block mit Gap, andere Aktivitäten → Gehen.
- Je strenger der Split, desto ehrlicher und meist niedriger das Ergebnis. _Eure Zahlen nach dem echten Lauf eintragen._

**17. Was ist der Cross-Activity-Test, und was beweist er nicht?**
- Training auf allen anderen Aktivitäten, Test auf Gehen (`cross_activity_split`). Der Test ist eine andere Aufnahme.
- Aber: derselbe Tag und derselbe Sensorsitz. Echte Wiedererkennung über Tage zeigt er nicht.

## D. Merkmale (`merkmale.py`)

**18. Welche Merkmale berechnet ihr?**
- Pro Kanal mean, std, min, max, rms, energie, wie in Abschnitt 5 vorgegeben. Dazu dieselben sechs für den Betrag von acc und gyr jedes Sensors.
- 24 Kanäle × 6 + 8 Beträge × 6 = 192 Merkmale pro Fenster.

**19. Was ist der Unterschied zwischen RMS und Energie?**
- Energie ist hier der Mittelwert der Quadrate, RMS die Wurzel daraus. Beide enthalten dieselbe Information.
- Beide sind drin, weil die Pipeline beide nennt. Für den Random Forest ist das egal; bei KNN zählt diese Information dadurch doppelt.

**20. Was misst der Mittelwert einer Beschleunigungsachse eigentlich?**
- Vor allem die Schwerkraft in Sensorkoordinaten, also wie der Sensor gedreht sitzt. Die Bewegung mittelt sich über mehrere Schritte weitgehend heraus.
- Beim Gyroskop zeigt der Mittelwert vor allem den Offset des Geräts.

**21. Warum der Betrag?**
- |a| = Wurzel aus x² + y² + z² ist die Länge des Vektors. Sie ändert sich nicht, wenn man den Sensor dreht.
- `test_betrag_haengt_nicht_von_der_drehung_ab` dreht die Daten zufällig und prüft genau das.

## E. Modelle (`modelle.py`)

**22. Welche Modelle, und warum?**
- Abschnitt 7 nennt Random Forest, SVM und KNN, optional ein 1D-CNN, dazu eine Baseline. Wir haben Mehrheits-Baseline, Entscheidungsbaum, KNN, SVM und Random Forest.
- Von einfach und gut erklärbar bis stark. Alle laufen auf denselben Folds, sonst wäre der Vergleich unfair.

**23. Wie funktioniert ein Random Forest?**
- Viele Entscheidungsbäume (bei uns 300). Jeder lernt auf einer zufälligen Stichprobe der Trainingsdaten und prüft an jedem Knoten nur eine zufällige Auswahl von Merkmalen.
- Die Vorhersage ist der Mittelwert der Bäume. Dadurch passt er sich weniger an Zufälligkeiten an als ein einzelner Baum.

**24. Wie arbeiten KNN und SVM, und warum brauchen sie Skalierung?**
- KNN: Die 5 ähnlichsten Trainingsfenster (kleinster Abstand) stimmen ab.
- SVM: Sucht Trennflächen mit möglichst großem Abstand zu den Klassen; der RBF-Kernel erlaubt gekrümmte Grenzen, C = 10 regelt, wie streng Fehler bestraft werden.
- Beide rechnen mit Abständen. Ohne Skalierung würden Merkmale mit großen Zahlen dominieren (Gyroskop in Grad pro Sekunde, um 100) gegenüber kleinen (Beschleunigung in g, um 1).

**25. Warum steckt der StandardScaler in einer Pipeline?**
- Damit Mittelwert und Streuung nur aus den Trainingsdaten gelernt werden (Abschnitt 4). Würde man vorher alle Daten skalieren, flösse Information aus dem Test ein.

**26. Wozu `CalibratedClassifierCV` bei der SVM?**
- Eine SVM gibt nur Abstände zur Trennfläche aus, keine Wahrscheinlichkeiten. Für die Clip-Entscheidung brauchen wir Wahrscheinlichkeiten.
- Die Kalibrierung (Platt-Skalierung) bildet die Abstände mit einer logistischen Funktion auf Werte zwischen 0 und 1 ab.

**27. Warum habt ihr keine Hyperparameter optimiert?**
- Wählt man Parameter nach dem besten Testergebnis, ist das selbst eine Form von Leakage. Richtig wäre eine verschachtelte Kreuzvalidierung.
- Bei der kleinen Datenmenge nutzen wir übliche Startwerte und vergleichen die Modelle fair mit denselben Folds.

## F. Bewertung

**28. Wie wird aus den Fenstern eine Entscheidung für den Clip?**
- Für alle Fenster des Clips werden die Wahrscheinlichkeiten jeder Person gemittelt; der höchste Mittelwert gewinnt (`clip_entscheidung`).
- Einzelne unsichere Fenster fallen dadurch weniger ins Gewicht. Deshalb ist die Clip-Ebene meist besser als die Fenster-Ebene.

**29. Warum Macro-F1?**
- F1 ist das harmonische Mittel aus Precision und Recall. Macro-F1 berechnet es für jede Person und mittelt dann, sodass jede Person gleich viel zählt.
- Accuracy kann bei ungleich vielen Fenstern pro Person zu gut aussehen. Macro-F1 ist die zentrale Kennzahl der Pipeline (Abschnitt 5).

**30. Wo liegt das Zufallsniveau?**
- Bei n Personen etwa 1/n, bei 25 Personen also 4 %. Die gestrichelte Linie in den Grafiken.
- Die Mehrheits-Baseline sagt immer dieselbe Person: Accuracy etwa 1/n, Macro-F1 fast 0.

**31. Was sagt Mittelwert ± Standardabweichung über die Folds?**
- 5 Folds ergeben 5 Ergebnisse; die Streuung zeigt, wie stabil das Ergebnis ist (Abschnitt 5).
- Vorsicht: Die Folds sind nicht unabhängig (dieselben Personen, dieselben Aufnahmen). Die Streuung ist nur ein grober Hinweis.

**32. Was zeigt die Confusion Matrix?**
- Zeilen sind die wahren Personen, Spalten die vorhergesagten, die Diagonale ist richtig. Man sieht, welche Personen verwechselt werden.

## G. Ergebnisse und Grenzen

**33. Was ist euer wichtigstes Ergebnis?**
- _Nach dem Lauf auf echten Daten ausfüllen: bestes Modell (E1), Abfall vom Block-Split zu Cross-Activity (E2), Lage-Test (E3), bester Sensor (E4)._

**34. Was zeigt der Lage-Test (E3)?**
- Ist „nur_mittelwerte“ fast so gut wie „alle“, reicht der Sensorsitz zur Unterscheidung. Das Session-Risiko ist dann real.
- „nur_betraege“ zeigt, was ohne Lageinformation übrig bleibt. Mit Cross-Activity sieht man, was sich auf eine andere Aufnahme überträgt.

**35. Was würdet ihr als Nächstes tun?**
- Einige Personen an einem zweiten Tag messen und Tag 1 → Tag 2 testen (Cross-Session).
- Mehr Personen; 1D-CNN aus der Pipeline; Fensterlänge und Anzahl Clips variieren.

## H. Datenschutz und Ethik

**36. Warum ist Personenerkennung aus Gangdaten heikel?**
- Der Gang kann eine Person identifizieren; das kann biometrisch sein (DSGVO Art. 4 Nr. 14, bei Identifizierung Art. 9).
- Deshalb nur pseudonymisierte IDs, nur im Rahmen der Einwilligung (Abschnitt 10), keine Messdaten und keine trainierten Modelle im öffentlichen Repo.

## I. Arbeitsweise

**37. Wer hat was gemacht?**
- Siehe `CONTRIBUTIONS.md`. Jede Person kennt ihren Teil im Detail und den Rest im Überblick.

**38. Habt ihr KI-Werkzeuge genutzt?**
- Ehrlich und genau so, wie es im README unter „Entstehung und KI-Nutzung“ steht: wofür, und was ihr selbst gemacht, geprüft und entschieden habt.

---

## Wenn ihr etwas nicht wisst

Nicht raten. Sagt, was ihr sicher wisst, und wie ihr den Rest herausfinden würdet, zum Beispiel:
„Das würde ich prüfen, indem ich `GAP_S` in `config.py` auf 0 setze und E2 vergleiche.“
