# Women's Health Clustering & Personalized Recommendations

## 📌 Overview
PCOSaathi is an AI-powered health companion designed to support women in managing PCOS (Polycystic Ovary Syndrome) by providing personalized health recommendations. Submitted under the **Infosys Springboard iAccelerate Women’s Hackathon 2025**, this project uses machine learning to analyze women's health data and offer insights tailored to Indian dietary habits and lifestyle preferences.

The application leverages **machine learning** to analyze women's health data and provide **personalized health recommendations**.It clusters users based on key health parameters like **weight gain, hair loss, acne, period regularity, and exercise** to generate insights for managing conditions like **PCOS (Polycystic Ovary Syndrome)**.

The application is built using **Streamlit** for the frontend, with **Agglomerative Clustering** and **Gaussian Mixture Models (GMM)** for clustering. The recommendations are tailored to cluster women based on their health needs and prefrences in context of Indian dietary habits and lifestyle preferences.

## 🚀 Features
- **Data-driven clustering**: Uses machine learning models to group users based on health parameters, needs and prefrences.
- **Personalized recommendations**: Offers customized lifestyle, diet, and exercise advice.
- **Indian context**: Tailored suggestions based on local food, routines, and wellness practices.
- **User-friendly UI**: Built with Streamlit for easy interaction.

## 🏗 Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python (Pandas, NumPy, Scikit-Learn)
- **Visualization**: Seaborn, Matplotlib
- **Machine Learning Models**:
  - Agglomerative Clustering (Hierarchical Clustering)
  - Gaussian Mixture Model (GMM)
- **Data Processing**: StandardScaler for normalization

## 📂 Project Structure
```
├── Women-Health-Clustering/
│   ├── app.py                    # Main Streamlit application
│   ├── requirements.txt          # Python dependencies
|   ├── myData.py                 #Jupyter Notebook
│   ├── dataset/
│   │   ├── CLEAN-PCOS SURVEY SPREADSHEET.csv    # Cleaned dataset
│   ├── README.md                  # Project documentation
```

## ⚡ Installation & Setup
1. **Clone the repository**
```bash
   git clone https://github.com/yourusername/Women-Health-Clustering.git
   cd Women-Health-Clustering
```

2. **Create a virtual environment** (Recommended)
```bash
   python -m venv venv
   source venv/bin/activate  # On Mac/Linux
   venv\Scripts\activate     # On Windows
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Run the application**
```bash
   streamlit run app.py
```

## 🎯 How It Works
1. **User inputs their health data**: Age, weight gain, hair loss, acne severity, period regularity, and exercise routine.
2. **Machine learning models analyze the data**: Clusters users based on similarities.
3. **Personalized health recommendations**: The app provides practical, culturally relevant health advice.

## Archetecture Diagram
| ![Diagram](C:/Users/jyots/OneDrive/Desktop/Women-Health/assets/architecture_women.jpg) |

## FlowChart
| ![FlowChart](C:/Users/jyots/OneDrive/Desktop/Women-Health/women-flowchart.png) |

## 🖼 Screenshots
| Input Section | Recommendation Output |
|-----------|--------------|-----------------------|
| ![Input](C:/Users/jyots/OneDrive/Desktop/Women-Health/assets/Pic_1.png) | ![Output](C:/Users/jyots/OneDrive/Desktop/Women-Health/assets/Pic_2.png) |

## 🎥 Demo Video
Watch the full demo on YouTube: [YouTube Link Here](https://youtu.be/example)

## 🔥 Contributing
Want to improve this project? Follow these steps:
1. **Fork** the repository
2. **Create a new branch** (`feature-new-improvement`)
3. **Make changes** and commit (`git commit -m 'Add a cool feature'`)
4. **Push to your branch** and submit a **Pull Request**

## 📜 License
This project is licensed under the **MIT License**.

---

🌸 *Empowering women's health through data-driven insights!*

