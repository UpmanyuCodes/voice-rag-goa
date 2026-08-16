"""Curated Multilingual Samples from ai4bharat/MSMARCO-XI.

Provides grounded queries, passages with relevance markers (is_selected),
and answers across Hindi, Bengali, Tamil, Telugu, Marathi, and English.
"""

from typing import List, Dict, Any

SAMPLE_RECORDS: List[Dict[str, Any]] = [
    {
        "query_id": 10001,
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "language": "hi",
        "Eng_Query": "what are the main symptoms of malaria?",
        "query": "मलेरिया के मुख्य लक्षण क्या हैं?",
        "Eng_Answer": "The main symptoms of malaria include high fever, shaking chills, profuse sweating, headache, nausea, vomiting, muscle pain, and extreme fatigue.",
        "Answer": "मलेरिया के मुख्य लक्षणों में तेज बुखार, कंपकंपी के साथ ठंड लगना, अत्यधिक पसीना आना, सिरदर्द, मतली, उल्टी, मांसपेशियों में दर्द और अत्यधिक थकान शामिल हैं।",
        "passages": {
            "is_selected": [1, 0, 0],
            "English_passages": [
                "Malaria signs and symptoms typically begin within a few weeks after being bitten by an infected mosquito. Common symptoms include high fever, severe shaking chills, profuse sweating as body temperature falls, persistent headache, nausea, vomiting, muscle body aches, diarrhea, and general malaise.",
                "Malaria is caused by single-celled microorganisms of the Plasmodium group. It is spread exclusively through the bites of infected female Anopheles mosquitoes.",
                "Preventative medication for travelers includes chloroquine, atovaquone-proguanil, and doxycycline depending on geographic drug resistance patterns."
            ],
            "Translated_passages": [
                "मलेरिया के लक्षण और संकेत आमतौर पर संक्रमित मच्छर के काटने के कुछ हफ्तों के भीतर शुरू होते हैं। सामान्य लक्षणों में तेज बुखार, गंभीर कंपकंपी, बुखार उतरते समय अत्यधिक पसीना, लगातार सिरदर्द, मतली, उल्टी, मांसपेशियों में दर्द, दस्त और सामान्य अस्वस्थता शामिल हैं।",
                "मलेरिया प्लास्मोडियम समूह के एकल-कोशिका वाले सूक्ष्मजीवों के कारण होता है। यह विशेष रूप से संक्रमित मादा एनोफिलीज मच्छरों के काटने से फैलता है।",
                "यात्रियों के लिए निवारक दवाओं में भौगोलिक दवा प्रतिरोध पैटर्न के आधार पर क्लोरोक्वीन, एटोवाक्वोन-प्रोगुआनिल और डॉक्सीसाइक्लिन शामिल हैं।"
            ]
        }
    },
    {
        "query_id": 10002,
        "query_type": "ENTITY",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "language": "hi",
        "Eng_Query": "what is the capital of Goa and what is it known for?",
        "query": "गोवा की राजधानी क्या है और यह किसके लिए प्रसिद्ध है?",
        "Eng_Answer": "The capital of Goa is Panaji (Panjim), situated on the banks of the Mandovi river, renowned for its Portuguese colonial architecture, Fontainhas Latin Quarter, and cultural heritage.",
        "Answer": "गोवा की राजधानी पणजी (पंजिम) है, जो मांडवी नदी के तट पर स्थित है और अपनी पुर्तगाली औपनिवेशिक वास्तुकला, फोंटेनहास लैटिन क्वार्टर और सांस्कृतिक विरासत के लिए जानी जाती है।",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "Panaji, also known as Panjim, serves as the state capital of Goa, India. Located on the banks of the Mandovi estuary on the island of Tiswadi, it is celebrated for its colorful Portuguese colonial style villas, stone-paved streets, churches like Our Lady of the Immaculate Conception, and the historic Latin Quarter of Fontainhas.",
                "Goa is India's smallest state by area and the fourth-smallest by population. Vasco da Gama is the largest city in Goa."
            ],
            "Translated_passages": [
                "पणजी, जिसे पंजिम के नाम से भी जाना जाता है, भारत के गोवा राज्य की राजधानी है। तिसवाड़ी द्वीप पर मांडवी नदी के मुहाने पर स्थित, यह अपनी रंगीन पुर्तगाली औपनिवेशिक शैली के विला, पत्थर की पक्की सड़कों, अवर लेडी ऑफ द इमैक्युलेट कॉन्सेप्शन जैसे चर्चों और फोंटेनहास के ऐतिहासिक लैटिन क्वार्टर के लिए प्रसिद्ध है।",
                "गोवा क्षेत्रफल के हिसाब से भारत का सबसे छोटा और जनसंख्या के हिसाब से चौथा सबसे छोटा राज्य है। वास्को डी गामा गोवा का सबसे बड़ा शहर है।"
            ]
        }
    },
    {
        "query_id": 10003,
        "query_type": "NUMERIC",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "language": "hi",
        "Eng_Query": "what is the speed of light in vacuum?",
        "query": "निर्वात में प्रकाश की गति क्या है?",
        "Eng_Answer": "The speed of light in a vacuum is exactly 299,792,458 meters per second (approximately 300,000 kilometers per second).",
        "Answer": "निर्वात में प्रकाश की गति ठीक 299,792,458 मीटर प्रति सेकंड (लगभग 300,000 किलोमीटर प्रति सेकंड) है।",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "The speed of light in vacuum, commonly denoted c, is a universal physical constant exactly equal to 299,792,458 metres per second (approx. 3.00 x 10^8 m/s or 186,282 miles per second).",
                "Light takes approximately 8 minutes and 20 seconds to travel from the surface of the Sun to Earth."
            ],
            "Translated_passages": [
                "निर्वात में प्रकाश की गति, जिसे आमतौर पर c से दर्शाया जाता है, एक सार्वभौमिक भौतिक स्थिरांक है जो ठीक 299,792,458 मीटर प्रति सेकंड (लगभग 3.00 x 10^8 मी/से या 186,282 मील प्रति सेकंड) के बराबर है।",
                "सूर्य की सतह से पृथ्वी तक प्रकाश को पहुँचने में लगभग 8 मिनट और 20 सेकंड का समय लगता है।"
            ]
        }
    },
    {
        "query_id": 10004,
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "ben_Beng",
        "language": "bn",
        "Eng_Query": "what causes solar eclipse?",
        "query": "সূর্যগ্রহণ কেন হয়?",
        "Eng_Answer": "A solar eclipse occurs when the Moon passes directly between the Earth and the Sun, blocking all or part of the Sun's light from reaching Earth.",
        "Answer": "সূর্যগ্রহণ ঘটে যখন চাঁদ পৃথিবী এবং সূর্যের ঠিক মাঝখানে আসে, যার ফলে সূর্যের আলো পৃথিবীতে পৌঁছাতে বাধা পায়।",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "A solar eclipse happens when the Moon passes between Earth and the Sun, thereby totally or partly obscuring the image of the Sun for a viewer on Earth. Such an alignment coincides with a new moon, indicating the Moon is closest to the ecliptic plane.",
                "A lunar eclipse occurs when the Moon moves into the Earth's shadow. This can occur only when the Sun, Earth, and Moon are exactly or very closely aligned with Earth between the other two."
            ],
            "Translated_passages": [
                "একটি সূর্যগ্রহণ ঘটে যখন চাঁদ পৃথিবী এবং সূর্যের মাঝখান দিয়ে যায়, ফলে পৃথিবীর একজন দর্শকের জন্য সূর্যের দৃশ্য সম্পূর্ণরূপে বা আংশিকভাবে অস্পষ্ট হয়ে যায়। এই অবস্থানটি অমাবস্যার সাথে মিলে যায়।",
                "চন্দ্রগ্রহণ ঘটে যখন চাঁদ পৃথিবীর ছায়ায় চলে যায়। এটি কেবল তখনই ঘটতে পারে যখন সূর্য, পৃথিবী এবং চাঁদ ঠিক বা খুব ঘনিষ্ঠভাবে একই সরলরেখায় অবস্থান করে।"
            ]
        }
    },
    {
        "query_id": 10005,
        "query_type": "DESCRIPTION",
        "source_lang": "eng_Latn",
        "target_lang": "tam_Taml",
        "language": "ta",
        "Eng_Query": "how does photosynthesis work in plants?",
        "query": "தாவரங்களில் ஒளிச்சேர்க்கை எவ்வாறு செயல்படுகிறது?",
        "Eng_Answer": "Photosynthesis is the biological process by which green plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of glucose.",
        "Answer": "ஒளிச்சேர்க்கை என்பது பசுமைத் தாவரங்கள் சூரிய ஒளி, நீர் மற்றும் கார்பன் டை ஆக்சைடைப் பயன்படுத்தி ஆக்ஸிஜன் மற்றும் குளுக்கோஸ் வடிவில் ஆற்றலை உருவாக்கும் ஒரு உயிரியல் செயல்முறையாகும்.",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy. Using chlorophyll in chloroplasts, carbon dioxide from the air and water from the soil are transformed into glucose and oxygen as a byproduct.",
                "Cellular respiration is a set of metabolic reactions that take place in the cells of organisms to convert biochemical energy from nutrients into adenosine triphosphate (ATP)."
            ],
            "Translated_passages": [
                "ஒளிச்சேர்க்கை என்பது தாவரங்கள் ஒளி ஆற்றலை வேதியியல் ஆற்றலாக மாற்ற பயன்படுத்தும் ஒரு செயல்முறையாகும். பசுங்கணிகங்களில் உள்ள பச்சையம் மூலம், காற்றில் உள்ள கார்பன் டை ஆக்சைடு மற்றும் மண்ணில் உள்ள நீர் ஆகியவை குளுக்கோஸ் மற்றும் ஆக்ஸிஜனாக மாற்றப்படுகின்றன.",
                "செல்லுலார் சுவாசம் என்பது ஊட்டச்சத்துக்களிலிருந்து உயிர்வேதியியல் ஆற்றலை ATP ஆக மாற்றும் வளர்சிதை மாற்ற வினைகளின் தொகுப்பாகும்."
            ]
        }
    },
    {
        "query_id": 10006,
        "query_type": "ENTITY",
        "source_lang": "eng_Latn",
        "target_lang": "hin_Deva",
        "language": "hi",
        "Eng_Query": "who is considered the father of computer science?",
        "query": "कंप्यूटर विज्ञान का जनक किसे माना जाता है?",
        "Eng_Answer": "Alan Turing is widely considered the father of modern computer science and artificial intelligence.",
        "Answer": "एलन ट्यूरिंग को व्यापक रूप से आधुनिक कंप्यूटर विज्ञान और कृत्रिम बुद्धिमत्ता (AI) का जनक माना जाता है।",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": [
                "Alan Turing was an English mathematician, computer scientist, logician, and cryptanalyst. Turing was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a model of a general-purpose computer.",
                "Charles Babbage is known as the 'father of the computer' for conceptualizing the first mechanical computer, the Analytical Engine, in the early 19th century."
            ],
            "Translated_passages": [
                "एलन ट्यूरिंग एक अंग्रेज गणितज्ञ, कंप्यूटर वैज्ञानिक, तर्कशास्त्री और क्रिप्टएनालिस्ट थे। ट्यूरिंग मशीन के साथ एल्गोरिदम और गणना की अवधारणाओं को औपचारिक रूप देकर ट्यूरिंग सैद्धांतिक कंप्यूटर विज्ञान के विकास में अत्यधिक प्रभावशाली थे।",
                "चार्ल्स बैबेज को 19वीं सदी की शुरुआत में पहले यांत्रिक कंप्यूटर, एनालिटिकल इंजन की अवधारणा तैयार करने के लिए 'कंप्यूटर का जनक' कहा जाता है।"
            ]
        }
    }
]
