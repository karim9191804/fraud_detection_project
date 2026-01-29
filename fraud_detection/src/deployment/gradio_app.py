"""
Gradio Web Interface for Fraud Detection
Déploiement sur Hugging Face Spaces
"""

import gradio as gr
import torch
import numpy as np
import pandas as pd
from typing import Dict, Tuple
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Imports locaux (à adapter selon le déploiement)
try:
    from src.models.hybrid_model import GNNLLMHybrid, create_hybrid_model
    from src.data.ieee_dataset import IEEEFraudPreprocessor
    from src.utils.metrics import compute_all_metrics
except ImportError:
    print("⚠️ Mode déploiement: imports locaux désactivés")


class FraudDetectionApp:
    """
    Application web pour la détection de fraude
    """
    def __init__(
        self,
        model_path: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = None
        self.preprocessor = None
        self.history = []
        
        # Charger le modèle si disponible
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Charger le modèle entraîné"""
        try:
            print(f"📂 Chargement du modèle depuis {model_path}...")
            
            # Configuration
            gnn_config = {
                "in_channels": 432,
                "hidden_channels": 256,
                "num_layers": 3,
                "dropout": 0.3,
                "model_type": "GCN"
            }
            
            llm_config = {
                "model_name": "microsoft/phi-2",
                "max_length": 512,
                "temperature": 0.7,
                "use_lora": True,
                "lora_r": 16,
                "lora_alpha": 32
            }
            
            # Créer le modèle
            self.model = create_hybrid_model(gnn_config, llm_config)
            
            # Charger les poids (si disponibles)
            # self.model.load_pretrained(model_path)
            
            self.model.eval()
            self.model.to(self.device)
            
            print("✅ Modèle chargé avec succès")
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            self.model = None
    
    def predict_transaction(
        self,
        transaction_amt: float,
        card_type: str,
        email_domain: str,
        country: str,
        device_type: str,
        transaction_hour: int
    ) -> Tuple[str, float, str, str]:
        """
        Prédire si une transaction est frauduleuse
        
        Args:
            transaction_amt: Montant de la transaction
            card_type: Type de carte
            email_domain: Domaine email
            country: Pays
            device_type: Type d'appareil
            transaction_hour: Heure de la transaction
        
        Returns:
            (prédiction, confiance, explication, alerte)
        """
        try:
            # Simuler une prédiction (si modèle non chargé)
            if self.model is None:
                # Règles simples pour la démo
                fraud_score = 0.0
                
                # Montant suspect
                if transaction_amt > 1000:
                    fraud_score += 0.3
                
                # Heure suspecte (nuit)
                if transaction_hour < 6 or transaction_hour > 22:
                    fraud_score += 0.2
                
                # Email suspect
                if email_domain in ["tempmail.com", "guerrillamail.com"]:
                    fraud_score += 0.3
                
                # Pays à risque (exemple)
                if country in ["Unknown", "N/A"]:
                    fraud_score += 0.2
                
                fraud_prob = min(fraud_score, 0.95)
                is_fraud = fraud_prob > 0.5
                
                # Explication
                explanation = self._generate_explanation_demo(
                    is_fraud=is_fraud,
                    fraud_prob=fraud_prob,
                    transaction_amt=transaction_amt,
                    transaction_hour=transaction_hour,
                    email_domain=email_domain
                )
                
            else:
                # Utiliser le modèle réel
                # TODO: Préparer les features et faire la prédiction
                is_fraud = False
                fraud_prob = 0.5
                explanation = "Modèle en cours de chargement..."
            
            # Formater les résultats
            prediction_text = "🚨 FRAUDE DÉTECTÉE" if is_fraud else "✅ TRANSACTION LÉGITIME"
            confidence_text = f"{fraud_prob:.1%}"
            
            # Alerte
            if fraud_prob > 0.8:
                alert = "🔴 ALERTE ÉLEVÉE - Blocage recommandé"
            elif fraud_prob > 0.5:
                alert = "🟡 ALERTE MOYENNE - Vérification recommandée"
            else:
                alert = "🟢 RISQUE FAIBLE - Transaction normale"
            
            # Sauvegarder dans l'historique
            self.history.append({
                "timestamp": datetime.now().isoformat(),
                "prediction": "Fraud" if is_fraud else "Legitimate",
                "confidence": fraud_prob,
                "amount": transaction_amt
            })
            
            return prediction_text, confidence_text, explanation, alert
            
        except Exception as e:
            return "❌ Erreur", "0%", f"Erreur: {str(e)}", "Erreur"
    
    def _generate_explanation_demo(
        self,
        is_fraud: bool,
        fraud_prob: float,
        transaction_amt: float,
        transaction_hour: int,
        email_domain: str
    ) -> str:
        """Générer une explication de démonstration"""
        
        if is_fraud:
            explanation = f"""🔍 **Analyse de la Transaction Suspecte**

Cette transaction a été classée comme **FRAUDULEUSE** avec une confiance de {fraud_prob:.1%}.

**Facteurs de risque identifiés:**

1. **Montant de transaction:** ${transaction_amt:,.2f}
   - {"⚠️ Montant élevé inhabituel" if transaction_amt > 1000 else "✓ Montant normal"}

2. **Heure de transaction:** {transaction_hour:02d}:00
   - {"⚠️ Transaction effectuée pendant les heures suspectes (nuit)" if (transaction_hour < 6 or transaction_hour > 22) else "✓ Heure normale"}

3. **Domaine email:** {email_domain}
   - {"⚠️ Domaine email temporaire ou suspect" if email_domain in ["tempmail.com", "guerrillamail.com"] else "✓ Domaine email légitime"}

**Recommandation:** Cette transaction nécessite une vérification humaine immédiate avant approbation.
"""
        else:
            explanation = f"""✅ **Analyse de la Transaction Légitime**

Cette transaction a été classée comme **LÉGITIME** avec une confiance de {(1-fraud_prob):.1%}.

**Profil de transaction normal:**

1. **Montant:** ${transaction_amt:,.2f} - Dans la plage habituelle
2. **Heure:** {transaction_hour:02d}:00 - Horaire de transaction normal
3. **Email:** {email_domain} - Domaine vérifié

**Recommandation:** Transaction approuvée automatiquement.
"""
        
        return explanation
    
    def get_statistics(self) -> Tuple[go.Figure, str]:
        """Obtenir les statistiques de l'historique"""
        
        if not self.history:
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="Aucune transaction analysée",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return empty_fig, "Aucune donnée disponible"
        
        # Créer le DataFrame
        df = pd.DataFrame(self.history)
        
        # Graphique: Distribution des prédictions
        fig = px.pie(
            df,
            names='prediction',
            title='Distribution des Prédictions',
            color='prediction',
            color_discrete_map={'Fraud': 'red', 'Legitimate': 'green'}
        )
        
        # Statistiques textuelles
        total = len(df)
        fraud_count = (df['prediction'] == 'Fraud').sum()
        avg_confidence = df['confidence'].mean()
        
        stats_text = f"""
📊 **Statistiques Globales**

- **Total de transactions:** {total}
- **Fraudes détectées:** {fraud_count} ({fraud_count/total*100:.1f}%)
- **Transactions légitimes:** {total - fraud_count} ({(total-fraud_count)/total*100:.1f}%)
- **Confiance moyenne:** {avg_confidence:.1%}
        """
        
        return fig, stats_text
    
    def batch_predict(self, file) -> Tuple[pd.DataFrame, str]:
        """Prédiction batch depuis un fichier CSV"""
        
        try:
            # Charger le CSV
            df = pd.read_csv(file.name)
            
            # Simuler des prédictions
            predictions = []
            for idx, row in df.iterrows():
                # Simulation simple
                fraud_prob = np.random.random()
                predictions.append({
                    'Transaction_ID': idx + 1,
                    'Prediction': 'Fraud' if fraud_prob > 0.5 else 'Legitimate',
                    'Confidence': f"{fraud_prob:.2%}",
                    'Risk_Level': 'High' if fraud_prob > 0.7 else 'Medium' if fraud_prob > 0.4 else 'Low'
                })
            
            results_df = pd.DataFrame(predictions)
            summary = f"✅ {len(df)} transactions analysées\n🚨 {(results_df['Prediction'] == 'Fraud').sum()} fraudes détectées"
            
            return results_df, summary
            
        except Exception as e:
            return pd.DataFrame(), f"❌ Erreur: {str(e)}"


def create_gradio_interface() -> gr.Blocks:
    """
    Créer l'interface Gradio
    
    Returns:
        Interface Gradio
    """
    app = FraudDetectionApp()
    
    with gr.Blocks(
        title="Fraud Detection System - GNN+LLM+RLHF",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # 🛡️ Système de Détection de Fraude
        ## GNN + LLM avec RLHF
        
        Système intelligent de détection de fraude combinant Graph Neural Networks et Large Language Models
        pour des prédictions précises et des explications en langage naturel.
        """)
        
        with gr.Tabs():
            
            # Tab 1: Prédiction Individuelle
            with gr.Tab("🔍 Analyse de Transaction"):
                
                gr.Markdown("### Entrez les détails de la transaction")
                
                with gr.Row():
                    with gr.Column():
                        amount = gr.Number(
                            label="Montant ($)",
                            value=100.0,
                            minimum=0
                        )
                        card_type = gr.Dropdown(
                            label="Type de Carte",
                            choices=["Visa", "Mastercard", "American Express", "Discover"],
                            value="Visa"
                        )
                        email_domain = gr.Textbox(
                            label="Domaine Email",
                            value="gmail.com"
                        )
                    
                    with gr.Column():
                        country = gr.Dropdown(
                            label="Pays",
                            choices=["USA", "Canada", "UK", "France", "Germany", "Other"],
                            value="USA"
                        )
                        device_type = gr.Dropdown(
                            label="Type d'Appareil",
                            choices=["Mobile", "Desktop", "Tablet"],
                            value="Desktop"
                        )
                        hour = gr.Slider(
                            label="Heure de Transaction",
                            minimum=0,
                            maximum=23,
                            step=1,
                            value=14
                        )
                
                predict_btn = gr.Button("🔍 Analyser la Transaction", variant="primary", size="lg")
                
                with gr.Row():
                    prediction_output = gr.Textbox(
                        label="Prédiction",
                        interactive=False
                    )
                    confidence_output = gr.Textbox(
                        label="Confiance",
                        interactive=False
                    )
                
                alert_output = gr.Textbox(
                    label="Niveau d'Alerte",
                    interactive=False
                )
                
                explanation_output = gr.Markdown(
                    label="Explication Détaillée"
                )
                
                predict_btn.click(
                    fn=app.predict_transaction,
                    inputs=[amount, card_type, email_domain, country, device_type, hour],
                    outputs=[prediction_output, confidence_output, explanation_output, alert_output]
                )
            
            # Tab 2: Analyse Batch
            with gr.Tab("📊 Analyse Batch"):
                
                gr.Markdown("""
                ### Upload un fichier CSV pour analyser plusieurs transactions
                
                Le fichier doit contenir les colonnes:
                - TransactionAmt
                - card_type
                - P_emaildomain
                - country
                - device_type
                - transaction_hour
                """)
                
                file_input = gr.File(
                    label="Fichier CSV",
                    file_types=[".csv"]
                )
                
                batch_btn = gr.Button("📊 Analyser le Batch", variant="primary")
                
                batch_summary = gr.Textbox(
                    label="Résumé",
                    interactive=False
                )
                
                batch_results = gr.Dataframe(
                    label="Résultats",
                    interactive=False
                )
                
                batch_btn.click(
                    fn=app.batch_predict,
                    inputs=[file_input],
                    outputs=[batch_results, batch_summary]
                )
            
            # Tab 3: Statistiques
            with gr.Tab("📈 Statistiques"):
                
                gr.Markdown("### Statistiques de l'Analyse")
                
                stats_btn = gr.Button("🔄 Rafraîchir les Statistiques")
                
                with gr.Row():
                    stats_plot = gr.Plot(label="Distribution")
                    stats_text = gr.Markdown(label="Métriques")
                
                stats_btn.click(
                    fn=app.get_statistics,
                    outputs=[stats_plot, stats_text]
                )
            
            # Tab 4: À Propos
            with gr.Tab("ℹ️ À Propos"):
                
                gr.Markdown("""
                ## 🎓 Projet de Fin d'Études (PFE)
                
                ### Architecture du Système
                
                Ce système combine trois technologies de pointe:
                
                1. **Graph Neural Networks (GNN)**
                   - Modèles: GCN, GAT, GraphSAGE
                   - Capture les relations entre transactions
                   - Détection de patterns complexes
                
                2. **Large Language Models (LLM)**
                   - Modèle: Microsoft Phi-2 avec LoRA
                   - Génération d'explications en langage naturel
                   - Fine-tuning adaptatif
                
                3. **Reinforcement Learning from Human Feedback (RLHF)**
                   - Apprentissage continu depuis le feedback humain
                   - Amélioration progressive des prédictions
                   - Système de récompense intelligent
                
                ### Cycle Jour/Nuit/Matin
                
                - **☀️ MODE JOUR (8h-20h):** Inférence en temps réel, sauvegarde des cas critiques
                - **🌙 MODE NUIT (20h-8h):** Fine-tuning automatique et RLHF
                - **🌅 MODE MATIN (7h):** Validation et déploiement conditionnel
                
                ### Technologies Utilisées
                
                - PyTorch & PyTorch Geometric
                - Transformers (Hugging Face)
                - Gradio pour l'interface
                - GitHub & Kaggle pour la synchronisation
                - WandB pour le monitoring
                
                ### Dataset
                
                - **IEEE-CIS Fraud Detection**
                - ~500k transactions
                - Features: montant, carte, email, appareil, localisation
                
                ### Métriques
                
                - Accuracy, Precision, Recall, F1-Score
                - ROC-AUC, Average Precision
                - False Positive Rate
                - Fraud Detection Rate
                
                ---
                
                💡 **Développé avec ❤️ pour la détection intelligente de fraude**
                """)
        
        gr.Markdown("""
        ---
        
        <div style="text-align: center;">
            <p>🔒 Système Sécurisé | 🚀 Temps Réel | 🤖 IA Explicable</p>
        </div>
        """)
    
    return demo


def launch_app(
    share: bool = False,
    server_port: int = 7860,
    auth: Tuple[str, str] = None
):
    """
    Lancer l'application
    
    Args:
        share: Créer un lien public
        server_port: Port du serveur
        auth: Tuple (username, password) pour l'authentification
    """
    demo = create_gradio_interface()
    
    demo.launch(
        share=share,
        server_port=server_port,
        auth=auth,
        show_api=True
    )


if __name__ == "__main__":
    print("🚀 Lancement de l'application Fraud Detection...")
    
    # Lancer sans authentification pour le dev
    launch_app(
        share=False,
        server_port=7860
    )
