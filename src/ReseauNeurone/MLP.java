package ReseauNeurone;
import java.io.*;
/**
  * @author hubert.cardot
 */
public class MLP {  // pg du MLP, reseau de neurones a retropropagation

    static int NbClasses=3, NbCaract=4, NbEx=50, NbExApprent=25;
    static int NbCouches=3, NbCaches=6, NbApprent=2000; 
    static int NbNeurones[]={NbCaract+1, NbCaches+1, NbClasses}; //+1 pour neurone fixe
    static Double data[][][] = new Double[NbClasses][NbEx][NbCaract];
    static Double poids[][][], N[][], S[][], coeffApprent=0.01, coeffSigmoide=2.0/3;
    
    private static Double fSigmoide(Double x)  {       // f()
    	return Math.tanh(coeffSigmoide*x); } 
    
    private static Double dfSigmoide(Double x) {       // df()
    	return coeffSigmoide/Math.pow(Math.cosh(coeffSigmoide*x),2); } 
    
    public static void main(String[] args) {
        System.out.println("Caches="+NbCaches+" App="+NbApprent+" coef="+coeffApprent);
        initialisation();
        apprentissage();
        evaluation();
    }   
    private static void initialisation() {
        lectureFichier(); 
        //Allocation et initialisation aleatoire des poids
        poids= new Double[NbCouches-1][][];
        for (int couche=0; couche<NbCouches-1; couche++) {
        	poids[couche] = new Double[NbNeurones[couche+1]][];
        	for (int i=0; i<NbNeurones[couche+1]; i++) {
        		poids[couche][i] = new Double[NbNeurones[couche]];
        		for (int j=0; j<NbNeurones[couche]; j++) {
        			poids[couche][i][j] = (Math.random()-0.5)/10; //dans [-0,05; +0,05[
        		}
        	}
        }
        //Allocation des etats internes N et des sorties S des neurones
        N = new Double[NbCouches][];
        S = new Double[NbCouches][];
        for (int couche=0; couche<NbCouches; couche++) {
        	N[couche] = new Double[NbNeurones[couche]];
        	S[couche] = new Double[NbNeurones[couche]];
        }
    }
    private static void apprentissage() {  
    	//---------- e faire
    }     
    private static void evaluation() {
        int classeTrouvee, Ok=0, PasOk=0;
        for(int i=0; i<NbClasses; i++) {
            for(int j=NbExApprent; j<NbEx; j++) { // parcourt les ex. de test
                //---------- e faire              // calcul des N et S des neurones
                classeTrouvee = 0;                // recherche max parmi les sorties RN
                //---------- e faire
                //System.out.println("classe "+i+" classe trouvee "+classeTrouvee);
                if (i==classeTrouvee) Ok++; else PasOk++;
            }
        }
        System.out.println("Taux de reconnaissance : "+(Ok*100./(Ok+PasOk)));
    }
    private static void propagation(Double X[]) {
    //---------- a faire
    }
    private static void retropropagation(int classe) {
    //---------- a faire
    }   
    private static void lectureFichier() {
        // lecture des donnees a partir du fichier iris.data
        String ligne, sousChaine;
        int classe=0, n=0;
        try {
             BufferedReader fic=new BufferedReader(new FileReader("iris.data"));
             while ((ligne=fic.readLine())!=null) {
                for(int i=0;i<NbCaract;i++) {
                    sousChaine = ligne.substring(i*NbCaract, i*NbCaract+3);
                    data[classe][n][i] = Double.parseDouble(sousChaine);
                    //System.out.println(data[classe][n][i]+" "+classe+" "+n);
                }
                if (++n==NbEx) { n=0; classe++; }
             }
        }
        catch (Exception e) { System.out.println(e.toString()); }
    }
}  //------------------fin classe MLP--------------------