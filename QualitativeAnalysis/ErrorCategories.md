# Category 1 — Semantic overlap confusion

## Cases:
* Threat, Insult, Stalking, Damage, or Extortion -> Bodily Injury
* Embezzlement (Other) -> Theft
* Fraud -> Theft
* Contempt -> Disobedience
* Homicide -> Bodily Injury
* Drug Trafficking Association (Law 11.343/06) -> Drug Trafficking (Law 11.343/06)
* Theft -> Robbery
* Threat, Insult, Stalking, Damage, or Extortion -> Violation of Urgent Protective Measures
* Recovery of Stolen Vehicle (Theft) -> Recovery of Stolen Vehicle
* Slander -> Defamation
* Robbery of Pedestrian -> Cell Phone Robbery
* Contempt -> Aggravated Resistance
* Disobedience -> Aggravated Resistance
* Robbery of Pedestrian -> Robbery Inside Vehicle
* Theft from Pedestrian -> Theft Inside Public Transport
* Robbery of Commercial Establishment -> Robbery of Pedestrian

## Description:
The model correctly recognizes the event domain, but fails to distinguish between legally related classes that frequently share vocabulary and narrative context.

---

# Category 2 — Generic-versus-specific subtype confusion

## Cases:
* Drug Law (Other) (Law 11.343/06) -> Drug Trafficking (Law 11.343/06)
* Drug Trafficking Association (Law 11.343/06) -> Drug Law (Other) (Law 11.343/06)
* Damage (Other) -> Trespassing
* Unintentional Bodily Injury (Other) (Law 9503/97) -> Unintentional Bodily Injury Caused by Vehicle Collision
* Robbery of Pedestrian -> Robbery (Other)
* Bodily Injury (Other) -> Bodily Injury Caused by Punches, Slaps, and Kicks
* Theft from Commercial Establishment -> Theft Inside Commercial Establishment
* Theft from Pedestrian -> Theft (Other)
* Fraud (Other) -> Credit Card Fraud
* Robbery of Commercial Establishment -> Robbery Inside Commercial Establishment
* Unintentional Bodily Injury (Other) (Law 9503/97) -> Unintentional Bodily Injury Caused by Run-Over
* Bodily Injury (Other) -> Physical Altercation
* Homicide by Firearm Projectile -> Attempted Homicide by Firearm Projectile
* Robbery Inside Vehicle -> Robbery (Other)
* Robbery of Pedestrian -> Robbery (Other)
* Resistance -> Aggravated Resistance
* Embezzlement (Other) -> Fraud
* Fraud (Other) -> Attempted Fraud (Other)
* Vehicle Robbery -> Motorcycle Robbery
* Slander -> Insult (Other)
* Robbery of Pedestrian -> Robbery (Other)
* Vehicle Theft -> Motorcycle Theft
* Rape -> Rape of a Vulnerable Person
* Defamation -> Insult (Other)
* Insult (Other) -> Insult Due to Prejudice
* Robbery Inside Commercial Establishment -> Robbery (Other)
* Theft from Pedestrian -> Cell Phone Theft

## Description:
The model correctly identifies the general nature of the offense, but fails to distinguish the specific subtype within the same criminal category.

---

# Category 3 — Operational workflow confusion

## Cases:
* roperty Crimes -> Vehicle Recovery, Administrative Acts, or Incident Reporting
* Traffic or Environmental Crimes -> Vehicle Recovery, Administrative Acts, or Incident Reporting
* Vehicle Recovery, Administrative Acts, or Incident Reporting -> Drug, Narcotics, and Firearm Possession Related
* Crimes Against Persons -> Vehicle Recovery, Administrative Acts, or Incident Reporting
* Vehicle Recovery, Administrative Acts, or Incident Reporting -> Trespassing, Disturbance, Damage, or Arbitrary Exercise of Rights
* Administrative Acts -> Incident Reporting
* Administrative Acts -> Vehicle Recovery
* Disappearance (Other) -> Found Missing Person
* Execution of Arrest Warrant -> Resulting from Preventive Detention by Police Unit
* Vehicle Recovery, Administrative Acts, or Incident Reporting -> Drug, Narcotics, and Firearm Possession Related
* Vehicle Recovery, Administrative Acts, or Incident Reporting -> Resistance, Contempt, or Disobedience
* Administrative Acts -> Vehicle Recovery
* Seizure of Narcotic Substance -> Possession of Drugs for Personal Use (Law 11.343/06)
* Receiving Stolen Goods -> Robbery
* Seizure of Narcotic Substance -> Drug Trafficking (Law 11.343/06)

## Description:
The model appears to understand the central event, but confuses different stages within the same operational or administrative workflow.

---

# Category 4 — Crimes coexistence confusion

## Cases:
* Crimes Against Persons -> Trespassing, Disturbance, Damage, or Arbitrary Exercise of Rights
* Crimes Against Persons -> Resistance, Contempt, or Disobedience
* Crimes Against Persons -> Traffic or Environmental Crimes
* Property Crimes -> Trespassing, Disturbance, Damage, or Arbitrary Exercise of Rights
* Crimes Against Persons -> Property Crimes
* Crimes Against Persons -> Drug, Narcotics, and Firearm Possession Related
* Property Crimes -> Resistance, Contempt, or Disobedience
* Traffic or Environmental Crimes -> Resistance, Contempt, or Disobedience
* Crimes Against Persons -> Drug, Narcotics, and Firearm Possession Related
* Property Crimes -> Traffic or Environmental Crimes
* Drug, Narcotics, and Firearm Possession Related -> Resistance, Contempt, or Disobedience
* Drug Trafficking Association (Law 11.343/06) -> Illegal Possession of Restricted Firearm
* Illegal Possession of Restricted Firearm -> Drug Trafficking (Law 11.343/06)
* Drug Law (Other) (Law 11.343/06) -> Illegal Possession of Restricted Firearm
* Property Crimes -> Drug, Narcotics, and Firearm Possession Related
* Violation of Urgent Protective Measures -> Bodily Injury

## Description:
These cases represent narratives that contain elements of multiple offenses occurring within the same criminal episode.