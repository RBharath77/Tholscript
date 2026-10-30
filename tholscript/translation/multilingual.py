"""Multi-language Indic translation & transliteration engine for THOLSCRIPT.

Supports epigraphic and modern translation from Old/Medieval Tamil & Brahmi into:
- English (en)
- Telugu (te / తెలుగు)
- Kannada (kn / ಕನ್ನಡ)
- Malayalam (ml / മലയാളം)
- Hindi (hi / हिन्दी)
- Sanskrit (sa / संस्कृतम्)
- Marathi (mr / मराठी)
- Bengali (bn / বাংলা)
- Gujarati (gu / ગુજરાતી)
- Odia (or / ଓଡ଼ିଆ)

Uses:
1. Lexical-epigraphic semantic transfer (shared Dravidian/Sanskritic epigraphic roots)
2. Brahmic Tholscript glyph slot transliteration across Indic Unicode blocks
3. Multi-language speech synthesizer integration
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from tholscript.data.lexicon import GLOSS

# Language metadata
LANGUAGES = {
    "en": {"name": "English", "native": "English", "voice": "en-US", "icon": "🇬🇧"},
    "te": {"name": "Telugu", "native": "తెలుగు", "voice": "te-IN", "icon": "🏛️"},
    "kn": {"name": "Kannada", "native": "ಕನ್ನಡ", "voice": "kn-IN", "icon": "🪨"},
    "ml": {"name": "Malayalam", "native": "മലയാളം", "voice": "ml-IN", "icon": "🌴"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "voice": "hi-IN", "icon": "📜"},
    "sa": {"name": "Sanskrit", "native": "संस्कृतम्", "voice": "hi-IN", "icon": "🕉️"},
    "mr": {"name": "Marathi", "native": "मराठी", "voice": "mr-IN", "icon": "🌿"},
    "bn": {"name": "Bengali", "native": "বাংলা", "voice": "bn-IN", "icon": "🪔"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી", "voice": "gu-IN", "icon": "🪙"},
    "or": {"name": "Odia", "native": "ଓଡ଼ିଆ", "voice": "or-IN", "icon": "🐚"},
}

# --------------------------------------------------------------------------- #
# Epigraphic Lexical Dictionaries (Dravidian & Indic Cognates)
# --------------------------------------------------------------------------- #
EPIGRAPHIC_LEXICON: Dict[str, Dict[str, str]] = {
    # Titles & Rulers
    "வேந்தன்": {
        "en": "the emperor", "te": "చక్రవర్తి", "kn": "ಚಕ್ರವರ್ತಿ", "ml": "ചക്രവർത്തി",
        "hi": "सम्राट", "sa": "सम्राट्", "mr": "सम्राट", "bn": "সম্রাট", "gu": "સમ્રાટ", "or": "ସମ୍ରାଟ"
    },
    "மன்னன்": {
        "en": "the king", "te": "రాజు", "kn": "ರಾಜ", "ml": "രാജാവ്",
        "hi": "राजा", "sa": "नृपतिः", "mr": "राजा", "bn": "রাজা", "gu": "રાજા", "or": "ରାଜା"
    },
    "அரசன்": {
        "en": "the king", "te": "రాజు", "kn": "ಅರಸ", "ml": "അരചൻ",
        "hi": "नरेश", "sa": "राजा", "mr": "राजे", "bn": "নরেশ", "gu": "નરેશ", "or": "ନରେଶ"
    },
    "வேள்": {
        "en": "the chief (Vel)", "te": "నాయకుడు (వేళ్)", "kn": "ನಾಯಕ (ವೇಳ್)", "ml": "വേൾ പ്രഭു",
        "hi": "सामंत / प्रमुख (वेल)", "sa": "अधिपतिः", "mr": "प्रमुख", "bn": "অধিপতি", "gu": "સરદાર", "or": "ନାୟକ"
    },
    "கோன்": {
        "en": "the lord", "te": "ప్రభువు", "kn": "ಒಡೆಯ", "ml": "കോൻ പ്രഭു",
        "hi": "प्रभु / स्वामी", "sa": "स्वामी", "mr": "स्वामी", "bn": "প্রভু", "gu": "સ્વામી", "or": "ପ୍ରଭୁ"
    },
    "அதிகன்": {
        "en": "the lord (Adigan)", "te": "అదిగన్ ప్రభువు", "kn": "ಅದಿಗನ್ ಒಡೆಯ", "ml": "അധികൻ പ്രഭു",
        "hi": "अदिगन प्रभु", "sa": "अधिकाधिपः", "mr": "अधिपती", "bn": "অধিরাজ", "gu": "અધિકાન", "or": "ଅଧିପତି"
    },
    "சேரன்": {
        "en": "the Chera king", "te": "చేర రాజు", "kn": "ಚೇರ ರಾಜ", "ml": "ചേര രാജാവ്",
        "hi": "चेर राजा", "sa": "चेरनरेशः", "mr": "चेर राजा", "bn": "চের রাজা", "gu": "ચેરા રાજા", "or": "ଚେର ରାଜା"
    },
    "பாண்டியன்": {
        "en": "the Pandya king", "te": "పాండ్య రాజు", "kn": "ಪಾಂಡ್ಯ ರಾಜ", "ml": "പാണ്ഡ്യ രാജാവ്",
        "hi": "पांड्य राजा", "sa": "पाण्ड्यनरेशः", "mr": "पांड्य राजा", "bn": "পাণ্ড্য রাজা", "gu": "પાંડ્ય રાજા", "or": "ପାଣ୍ଡ୍ୟ ରାଜା"
    },
    "பெருமகன்": {
        "en": "the noble one", "te": "మహాపురుషుడు", "kn": "ಮಹಾಪುರುಷ", "ml": "പെരുമകൻ",
        "hi": "महानुभाव", "sa": "महापुरुषः", "mr": "थोर पुरुष", "bn": "মহাপুরুষ", "gu": "મહાપુરુષ", "or": "ମହାପୁରୁଷ"
    },

    # Places & Kingdoms
    "மதுரை": {
        "en": "Madurai", "te": "మధురై", "kn": "ಮಧುರೈ", "ml": "മധുര",
        "hi": "मदुरै", "sa": "मधुरा", "mr": "मदुरै", "bn": "মাদুরাই", "gu": "મદુરાઈ", "or": "ମଦୁରାଇ"
    },
    "வஞ்சி": {
        "en": "Vanchi (Chera capital)", "te": "వంచి", "kn": "ವಂಚಿ", "ml": "വഞ്ചി",
        "hi": "वंची", "sa": "वञ्ची", "mr": "वंची", "bn": "বঞ্চি", "gu": "વંચી", "or": "ବଞ୍ଚି"
    },
    "கொங்கு": {
        "en": "Kongu country", "te": "కొంగు దేశం", "kn": "ಕೊಂಗು ನಾಡು", "ml": "കൊങ്ങ് നാട്",
        "hi": "कोंगु देश", "sa": "कोङ्गुदेशः", "mr": "कोंगु देश", "bn": "কোঙ্গু দেশ", "gu": "કોંગુ દેશ", "or": "କୋଙ୍ଗୁ ଦେଶ"
    },
    "காஞ்சி": {
        "en": "Kanchipuram", "te": "కాంచీపురం", "kn": "ಕಾಂಚೀಪುರ", "ml": "കാഞ്ചീപുരം",
        "hi": "कांचीपुरम", "sa": "काञ्चीपुरम्", "mr": "कांचीपुरम", "bn": "কাঞ্চীপুরম", "gu": "કાંચીપુરમ", "or": "କାଞ୍ଚୀପୁରମ"
    },
    "தஞ்சாவூர்": {
        "en": "Thanjavur", "te": "తంజావూరు", "kn": "ತಂಜಾವೂರು", "ml": "തഞ്ചാവൂർ",
        "hi": "तंजாவूर", "sa": "तञ्जावूरु", "mr": "तंजावर", "bn": "তাঞ্জাভুর", "gu": "તંજાવુર", "or": "ତଞ୍ଜାଭୁର"
    },

    # Sacred & Monumental Structures
    "கோயில்": {
        "en": "temple", "te": "దేవాలయం / గుడి", "kn": "ದೇವಾಲಯ / ದೇಗುಲ", "ml": "ക്ഷേത്രം / കോവിൽ",
        "hi": "मंदिर", "sa": "देवालयम् / मन्दिरम्", "mr": "मंदिर", "bn": "মন্দির", "gu": "મંદિર", "or": "ମନ୍ଦିର"
    },
    "கோவில்": {
        "en": "temple", "te": "దేవాలయం", "kn": "ದೇವಾಲಯ", "ml": "ക്ഷേത്രം",
        "hi": "मंदिर", "sa": "देवालयम्", "mr": "मंदिर", "bn": "মন্দির", "gu": "મંદિર", "or": "ମନ୍ଦିର"
    },
    "கல்வெட்டு": {
        "en": "stone inscription", "te": "శిలా శాసనం", "kn": "ಶಿಲಾ ಶಾಸನ", "ml": "ശിലാശാസനം",
        "hi": "शिलालेख", "sa": "शिलाशासनम्", "mr": "शिलालेख", "bn": "শিলালেখ", "gu": "શિલાલેખ", "or": "ଶିଳାଲେଖ"
    },
    "நடுகல்": {
        "en": "hero stone", "te": "వీరగల్లు", "kn": "ವೀರಗಲ್ಲು", "ml": "വീരക്കല്ല്",
        "hi": "वीर प्रस्तर", "sa": "वीरशिला", "mr": "वीरगळ", "bn": "বীরপ্রস্তর", "gu": "વીરસ્તંભ", "or": "ବୀରଶିଳା"
    },
    "மண்டபம்": {
        "en": "pillared hall", "te": "మండపం", "kn": "ಮಂಟಪ", "ml": "മണ്ഡപം",
        "hi": "मंडप", "sa": "मण्डपम्", "mr": "मंडप", "bn": "মণ্ডপ", "gu": "મંડપ", "or": "ମଣ୍ଡପ"
    },
    "மடம்": {
        "en": "monastery / matha", "te": "మఠం", "kn": "ಮಠ", "ml": "മഠം",
        "hi": "मठ", "sa": "मठम्", "mr": "मठ", "bn": "মঠ", "gu": "મઠ", "or": "ମଠ"
    },
    "கிணறு": {
        "en": "water well", "te": "బావి", "kn": "ಬಾವಿ", "ml": "കിണർ",
        "hi": "कुआँ", "sa": "कूपः", "mr": "विहीर", "bn": "কূপ", "gu": "કૂવો", "or": "କୂଅ"
    },
    "ஏரி": {
        "en": "water reservoir / lake", "te": "చెరువు", "kn": "ಕೆರೆ", "ml": "കുളം / തടാകം",
        "hi": "जलाशय / झील", "sa": "तडागः", "mr": "तलाव", "bn": "জলাশয়", "gu": "તળાવ", "or": "ପୋଖରୀ"
    },

    # Offerings & Endowments
    "கொடை": {
        "en": "endowment gift", "te": "దానం / కానుక", "kn": "ದಾನ / ಕೊಡುಗೆ", "ml": "ദാനം",
        "hi": "दान / उपहार", "sa": "दानम् / उपहारः", "mr": "दान", "bn": "দান", "gu": "દાન", "or": "ଦାନ"
    },
    "தானம்": {
        "en": "charitable donation", "te": "దానం", "kn": "ದಾನ", "ml": "ദാനം",
        "hi": "दान", "sa": "दानम्", "mr": "दान", "bn": "দান", "gu": "દાન", "or": "ଦାନ"
    },
    "பொன்": {
        "en": "gold", "te": "బంగారం", "kn": "ಚಿನ್ನ / ಹೊನ್ನು", "ml": "സ്വർണ്ണം / പൊന്ന്",
        "hi": "स्वर्ण / सोना", "sa": "सुवर्णम्", "mr": "सुवर्ण", "bn": "স্বর্ণ", "gu": "સોનું", "or": "ସ୍ୱର୍ଣ୍ଣ"
    },
    "நிலம்": {
        "en": "land / territory", "te": "భూమి", "kn": "ಭೂಮಿ / ಹೊಲ", "ml": "ഭൂമി / നിലം",
        "hi": "भूमि / खेत", "sa": "भूमिः / क्षेत्रम्", "mr": "जमीन", "bn": "ভূমি", "gu": "જમીન", "or": "ଭୂମି"
    },
    "நெல்": {
        "en": "paddy / grain", "te": "వరి ధాన్యం", "kn": "ಭತ್ತ", "ml": "നെല്ല്",
        "hi": "धान / अन्न", "sa": "धान्यम्", "mr": "धान्य", "bn": "ধান্য", "gu": "ડાંગર", "or": "ଧାନ"
    },
    "பசு": {
        "en": "sacred cow", "te": "ఆవు / గోవు", "kn": "ಹಸು / ಗೋವು", "ml": "പശു",
        "hi": "गाय / गो", "sa": "धेनुः / गौः", "mr": "गाय", "bn": "ধেনু", "gu": "ગાય", "or": "ଗାଈ"
    },
    "ஆடு": {
        "en": "sheep", "te": "గొర్రె", "kn": "ಕುರಿ", "ml": "ആട്",
        "hi": "भेड़", "sa": "मेषः", "mr": "मेंढी", "bn": "মেষ", "gu": "ઘેટું", "or": "ମେଣ୍ଢା"
    },
    "பணம்": {
        "en": "coins / currency", "te": "నాణేలు / ధనం", "kn": "ನಾಣ್ಯಗಳು / ಹಣ", "ml": "നാണയം / പണം",
        "hi": "मुद्रा / धन", "sa": "नाणकम् / धनम्", "mr": "नाणी", "bn": "মুদ্রা", "gu": "સિક્કા", "or": "ମୁଦ୍ରା"
    },

    # Actions / Verbs (Epigraphic past tense)
    "அளித்தான்": {
        "en": "granted / endowed", "te": "సమర్పించాడు / ఇచ్చాడు", "kn": "ಕೊಟ್ಟನು / ನೀಡಿದನು", "ml": "നൽകി",
        "hi": "प्रदान किया", "sa": "दत्तवान्", "mr": "प्रदान केले", "bn": "প্রদান করলেন", "gu": "આપ્યું", "or": "ପ୍ରଦାନ କଲେ"
    },
    "கொடுத்தான்": {
        "en": "donated", "te": "దానం చేశాడు", "kn": "ದಾನಮಾಡಿದನು", "ml": "കൊടുത്തു",
        "hi": "दान में दिया", "sa": "दत्तवान्", "mr": "दान दिले", "bn": "দান করলেন", "gu": "દાન કર્યું", "or": "ଦାନ କଲେ"
    },
    "கட்டினான்": {
        "en": "constructed / built", "te": "నిర్మించాడు", "kn": "ಕಟ್ಟಿಸಿದನು", "ml": "നിർമ്മിച്ചു / പണിതു",
        "hi": "निर्माण कराया", "sa": "निर्मितवान्", "mr": "बांधले", "bn": "নির্মাণ করলেন", "gu": "બંધાવ્યું", "or": "ନିର୍ମାଣ କଲେ"
    },
    "செய்தான்": {
        "en": "made / commissioned", "te": "చేశాడు", "kn": "ಮಾಡಿದನು", "ml": "ചെയ്തു",
        "hi": "बनाया", "sa": "कृतवान्", "mr": "केले", "bn": "করলেন", "gu": "બનાવ્યું", "or": "କଲେ"
    },
    "அமைத்தான்": {
        "en": "installed / founded", "te": "ప్రతిష్ఠించాడు / స్థాపించాడు", "kn": "ಸ್ಥಾಪಿಸಿದನು", "ml": "സ്ഥാപിച്ചു",
        "hi": "स्थापित किया", "sa": "प्रतिष्ठापितवान्", "mr": "स्थापना केली", "bn": "স্থাপন করলেন", "gu": "સ્થાપના કરી", "or": "ସ୍ଥାପନ କଲେ"
    },
    "குடைந்தான்": {
        "en": "rock-excavated / chiseled", "te": "శిలలో చెక్కించాడు / గుహ తవ్వించాడు", "kn": "ಬಂಡೆಯಲ್ಲಿ ಕೊರೆಸಿದನು", "ml": "പാറ തുരന്നു പണിതു",
        "hi": "शिला को तराश कर बनाया", "sa": "उत्कीर्णवान्", "mr": "पाषाणात कोरले", "bn": "পাথরে খোদাই করলেন", "gu": "પથ્થર કોતરી બનાવ્યું", "or": "ପଥର ଖୋଦେଇ କଲେ"
    },
    "எழுதினான்": {
        "en": "inscribed / engraved", "te": "చెక్కించాడు / వ్రాశాడు", "kn": "ಬರೆಸಿದನು", "ml": "എഴുതിച്ചു",
        "hi": "अंकित कराया", "sa": "लिखितवान्", "mr": "लिहिले", "bn": "লিপিবদ্ধ করলেন", "gu": "લખાવ્યું", "or": "ଲେଖାଇଲେ"
    },
    "பாதுகாத்தான்": {
        "en": "protected / preserved", "te": "రక్షించాడు", "kn": "ರಕ್ಷಿಸಿದನು", "ml": "സംരക്ഷിച്ചു",
        "hi": "संरक्षित किया", "sa": "रक्षितवान्", "mr": "रक्षण केले", "bn": "রক্ষা করলেন", "gu": "રક્ષણ કર્યું", "or": "ରକ୍ଷା କଲେ"
    },

    # Kinship & Social Relations
    "மகன்": {
        "en": "son", "te": "కుమారుడు", "kn": "ಮಗ", "ml": "മകൻ",
        "hi": "पुत्र / बेटा", "sa": "सुतः / पुत्रः", "mr": "मुलगा", "bn": "পুত্র", "gu": "પુત્ર", "or": "ପୁଅ"
    },
    "மகள்": {
        "en": "daughter", "te": "కుమార్తె", "kn": "ಮಗಳು", "ml": "മകൾ",
        "hi": "पुत्री / बेटी", "sa": "सुता / कन्या", "mr": "मुलगी", "bn": "কন্যা", "gu": "પુત્રી", "or": "ଝିଅ"
    },
    "மனைவி": {
        "en": "consort / wife", "te": "భార్య / దేవి", "kn": "ಪತ್ನಿ / ಹೆಂಡತಿ", "ml": "ഭാര്യ / പത്നി",
        "hi": "पत्नी / रानी", "sa": "भार्या / धर्मपत्नी", "mr": "पत्नी", "bn": "স্ত্রী", "gu": "પત્ની", "or": "ପତ୍ନୀ"
    },
    "தம்பி": {
        "en": "younger brother", "te": "తమ్ముడు", "kn": "ತಮ್ಮ", "ml": "അനുജൻ",
        "hi": "छोटा भाई", "sa": "अनुजः", "mr": "लहान भाऊ", "bn": "ছোট ভাই", "gu": "નાનો ભાઈ", "or": "ସାନ ଭାଇ"
    },
    "அண்ணன்": {
        "en": "elder brother", "te": "అన్న", "kn": "ಅಣ್ಣ", "ml": "ജ്യേഷ്ഠൻ",
        "hi": "बड़ा भाई", "sa": "अग्रजः", "mr": "मोठा भाऊ", "bn": "বড় ভাই", "gu": "મોટો ભાઈ", "or": "ବଡ଼ ଭାଇ"
    },
    "தந்தை": {
        "en": "father", "te": "తండ్రి", "kn": "ತಂದೆ", "ml": "പിതാവ്",
        "hi": "पिता", "sa": "पिता / जनकः", "mr": "वडील", "bn": "পিতা", "gu": "પિતા", "or": "ପିତା"
    },
    "தாய்": {
        "en": "mother", "te": "తల్లి", "kn": "ತಾಯಿ", "ml": "മാതാവ്",
        "hi": "माता", "sa": "माता / जननी", "mr": "आई", "bn": "মাতা", "gu": "માતા", "or": "ମାତା"
    },

    # Epigraphic Formulae & Commemorations
    "வாழ்க": {
        "en": "may live long / victory be", "te": "వర్ధిల్లుగాక / జయమగుగాక", "kn": "ಬಾಳಲಿ / ಜಯವಾಗಲಿ", "ml": "വാഴട്ടെ / ജയിക്കട്ടെ",
        "hi": "दीर्घायु हों / जय हो", "sa": "विजयताम् / चिरञ्जीवतु", "mr": "दीर्घायुष्य लाभो", "bn": "দীর্ঘজীবী হোন", "gu": "ચિરંજીવી રહો", "or": "ଜୟ ହେଉ"
    },
    "ஆண்டு": {
        "en": "regnal year", "te": "పరిపాలనా సంవత్సరం", "kn": "ಆಳ್ವಿಕೆಯ ವರ್ಷ", "ml": "ഭരണ വർഷം",
        "hi": "शासन वर्ष", "sa": "राज्यवर्षम्", "mr": "राज्य वर्ष", "bn": "রাজত্ব বর্ষ", "gu": "રાજ્ય વર્ષ", "or": "ଶାସନ ବର୍ଷ"
    },
    "போர்": {
        "en": "battle / war", "te": "యుద్ధం", "kn": "ಯುದ್ಧ / ಕಾಳಗ", "ml": "യുദ്ധം",
        "hi": "युद्ध / संग्राम", "sa": "सङ्ग्रामः / युद्धम्", "mr": "युद्ध", "bn": "যুদ্ধ", "gu": "યુદ્ધ", "or": "ଯୁଦ୍ଧ"
    },
    "வீரன்": {
        "en": "valiant warrior", "te": "వీరుడు", "kn": "ವೀರ", "ml": "വീരൻ",
        "hi": "वीर योद्धा", "sa": "वीरः / भटः", "mr": "वीर", "bn": "বীর যোদ্ধা", "gu": "વીર", "or": "ବୀର"
    },
    "இறந்த": {
        "en": "who attained martyrdom / fell", "te": "వీరమరణం పొందిన", "kn": "ವೀರಮರಣ ಹೊಂದಿದ", "ml": "വീരമൃത്യു വരിച്ച",
        "hi": "वीरगति को प्राप्त", "sa": "वीरगतिं गतः", "mr": "वीरगती प्राप्त", "bn": "বীরগতি প্রাপ্ত", "gu": "વીરગતિ પામેલ", "or": "ବୀରଗତି ପ୍ରାପ୍ତ"
    },
    "இந்த": {
        "en": "this", "te": "ఈ", "kn": "ಈ", "ml": "ഈ",
        "hi": "यह", "sa": "अयम् / इदम्", "mr": "हा / हे", "bn": "এই", "gu": "આ", "or": "ଏହି"
    },
    "அந்த": {
        "en": "that", "te": "ఆ", "kn": "ಆ", "ml": "ആ",
        "hi": "वह", "sa": "तत् / सः", "mr": "तो / ते", "bn": "সেই", "gu": "તે", "or": "ସେହି"
    },
}

# --------------------------------------------------------------------------- #
# Unicode Indic Script Equivalence Transliteration
# --------------------------------------------------------------------------- #
# Unicode standard Indic script slots align relative to block base:
# Devanagari = 0x0900, Bengali = 0x0980, Gujarati = 0x0A80, Odia = 0x0B00,
# Tamil = 0x0B80, Telugu = 0x0C00, Kannada = 0x0C80, Malayalam = 0x0D00.

# Tamil Character to Phonetic Token mapping
_TAMIL_TO_INDIC_SLOT = {
    # Vowels
    "அ": 0x05, "ஆ": 0x06, "இ": 0x07, "ஈ": 0x08, "உ": 0x09, "ஊ": 0x0A,
    "எ": 0x0E, "ஏ": 0x0F, "ஐ": 0x10, "ஒ": 0x12, "ஓ": 0x13, "ஔ": 0x14,
    # Consonants
    "க": 0x15, "ங": 0x19, "ச": 0x1A, "ஞ": 0x1E, "ட": 0x1F, "ண": 0x23,
    "த": 0x24, "ந": 0x28, "ப": 0x2A, "ம": 0x2E, "ய": 0x2F, "ர": 0x30,
    "ல": 0x32, "வ": 0x35, "ழ": 0x34, "ள": 0x33, "ற": 0x31, "ன": 0x29,
    # Matras (Vowel signs)
    "\u0BBE": 0x3E, "\u0BBF": 0x3F, "\u0BC0": 0x40, "\u0BC1": 0x41, "\u0BC2": 0x42,
    "\u0BC6": 0x46, "\u0BC7": 0x47, "\u0BC8": 0x48, "\u0BCA": 0x4A, "\u0BCB": 0x4B,
    "\u0BCC": 0x4C, "\u0BCD": 0x4D, # Virama / Pulli
}

SCRIPT_OFFSETS = {
    "te": 0x0C00,  # Telugu
    "kn": 0x0C80,  # Kannada
    "ml": 0x0D00,  # Malayalam
    "hi": 0x0900,  # Devanagari (Hindi)
    "sa": 0x0900,  # Devanagari (Sanskrit)
    "mr": 0x0900,  # Devanagari (Marathi)
    "bn": 0x0980,  # Bengali
    "gu": 0x0A80,  # Gujarati
    "or": 0x0B00,  # Odia
}


def transliterate_tamil_to_indic(tamil_text: str, target_script: str) -> str:
    """Transliterates Tamil script to any Indic Brahmic script (Telugu, Kannada, etc.)."""
    if target_script == "en":
        from tholscript.translation.brahmi_map import tamil_to_roman
        return " ".join(tamil_to_roman(w) for w in tamil_text.split())

    offset = SCRIPT_OFFSETS.get(target_script)
    if not offset:
        return tamil_text

    out = []
    for ch in tamil_text:
        slot = _TAMIL_TO_INDIC_SLOT.get(ch)
        if slot is not None:
            # Special adjustments for scripts that don't have short e/o (Devanagari, Bengali)
            if target_script in ("hi", "sa", "mr", "bn", "gu") and slot in (0x0E, 0x12):
                slot = slot + 1  # map to long E/O
            if target_script in ("hi", "sa", "mr", "bn", "gu") and slot in (0x46, 0x4A):
                slot = slot + 1  # map to long matra E/O
            out.append(chr(offset + slot))
        else:
            out.append(ch)
    return "".join(out)


def translate_epigraphic_text(tamil_text: str, target_lang: str = "te") -> Dict[str, str]:
    """Translates epigraphic Tamil text into the target Indian language or English."""
    if not tamil_text or not tamil_text.strip():
        return {
            "target_lang": target_lang,
            "language_name": LANGUAGES.get(target_lang, {}).get("name", target_lang),
            "translated": "",
            "script_transliterated": "",
            "voice": LANGUAGES.get(target_lang, {}).get("voice", "en-US")
        }

    lang_meta = LANGUAGES.get(target_lang, LANGUAGES["te"])
    words = tamil_text.strip().split()
    translated_words = []

    for w in words:
        # Check epigraphic lexicon
        entry = EPIGRAPHIC_LEXICON.get(w)
        if entry and target_lang in entry:
            translated_words.append(entry[target_lang])
        elif target_lang == "en" and w in GLOSS:
            translated_words.append(GLOSS[w])
        else:
            # Fallback: transliterate proper nouns, names, and unmapped words into native script
            trans_word = transliterate_tamil_to_indic(w, target_lang)
            translated_words.append(trans_word)

    translated_sentence = " ".join(translated_words)
    full_script_transliteration = transliterate_tamil_to_indic(tamil_text, target_lang)

    return {
        "target_lang": target_lang,
        "language_name": f"{lang_meta['name']} ({lang_meta['native']})",
        "translated": translated_sentence,
        "script_transliterated": full_script_transliteration,
        "voice": lang_meta.get("voice", "en-US")
    }


def translate_all_languages(tamil_text: str) -> Dict[str, Dict]:
    """Pre-generates translations across all supported Indian languages and English."""
    result = {}
    for code in LANGUAGES.keys():
        result[code] = translate_epigraphic_text(tamil_text, target_lang=code)
    return result

