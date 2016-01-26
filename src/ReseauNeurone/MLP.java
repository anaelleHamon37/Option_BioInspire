package ReseauNeurone;
import java.io.*;
/**
 * @author Heldmaier Nicolas et Hamon Anaëlle
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
		for(int cpt = 0 ; cpt < NbApprent ; cpt ++)
			for(int i = 0 ; i < NbClasses ; i++)
				for(int j = 0 ; j < NbExApprent ; j++)
				{
					propagation(data[i][j]);
					retropropagation(i);
				}
	}
	
	private static void evaluation() {
		int classeTrouvee, Ok=0, PasOk=0;
		for(int i=0; i<NbClasses; i++) {
			for(int j=0; j<NbExApprent; j++) { // parcourt les ex. de test
				propagation(data[i][j]);           // calcul des N et S des neurones
				classeTrouvee = 0;                // recherche max parmi les sorties RN
				Double max = -2.;
				for(int classe = 0; classe < NbClasses ; classe++)
				{
					if(S[NbCouches-1][classe] > max)
					{
						max = S[NbCouches-1][classe];
						classeTrouvee = classe;
					}
				}
				System.out.println("classe "+i+" classe trouvee "+classeTrouvee);
				if (i==classeTrouvee) Ok++; else PasOk++;
			}
		}
		System.out.println("Taux de reconnaissance : "+(Ok*100./(Ok+PasOk)));
	}
	
	private static void propagation(Double X[]) {
		//première couche
		for(int i = 0 ; i< NbCaract ; i++)
		{
			N[0][i] = X[i];
			S[0][i] = fSigmoide(X[i]);
		}		
		// theta
		N[0][NbCaract] = 1.0;
		S[0][NbCaract] = fSigmoide(1.0);
		
		//couche cachées
		for(int couche = 1 ; couche<NbCouches -1 ; couche++)
		{
			for(int i = 0 ; i < NbNeurones[couche] -1 ; i++)	{
				double Ni = 0.;
				for (int j = 0 ; j < NbNeurones[couche - 1] ; j++)
				{
					Ni += poids[couche - 1][i][j] * S[couche - 1][j];
				}
				N[couche][i] = Ni;
				S[couche][i] = fSigmoide(Ni);				
			}
			N[couche][NbCaches] = 1.0;
			S[couche][NbCaches] = fSigmoide(1.0);	
		}
		
		//couche sortie
		for(int i = 0 ; i < NbNeurones[NbCouches -1] ; i++)
		{
			double Ni = 0.;
			for (int j = 0 ; j < NbNeurones[NbCouches - 2] ; j++)
			{
				Ni += poids[NbCouches - 2][i][j] * S[NbCouches - 2][j];
			}
			N[NbCouches -1][i] = Ni;
			S[NbCouches -1][i] = fSigmoide(Ni);			
		}
	}
	private static void retropropagation(int classe) {
		
		// tableau des deltas avec deltas[0] les deltas calculés de la couche 1
		Double deltas[][] = new Double[NbCouches -1][];

		// calcul des deltas
		
		// couche de sortie
		deltas[NbCouches - 2] = new Double[NbNeurones[NbCouches - 1]];
		for(int i = 0 ; i < NbNeurones[NbCouches - 1] ; i++)	{
			// si c'est la bonne classe : di = 1 sinon = -1
			deltas[NbCouches - 2][i] = (S[NbCouches - 1][i] - ((i==classe)?1:-1))*dfSigmoide(N[NbCouches - 1][i]);
		}
		
		// autres couches
		for(int couche = NbCouches -2 ; couche > 0 ; couche -- )	{
			deltas[couche - 1] = new Double[NbNeurones[couche]];
			for(int i = 0 ; i < NbNeurones[couche] ; i++)	{
				double delta = 0.;
				for(int k = 0 ; k < NbNeurones[couche + 1] ; k++)
					delta += deltas[couche][k] * poids[couche][k][i];
				delta *= dfSigmoide(N[couche][i]);
				deltas[couche - 1][i] = delta;
			}			
		}
		
		// affectation des nouveaux poids
		for( int couche = NbCouches -1 ; couche > 0 ; couche --)
		{
			for(int i = 0 ; i < NbNeurones[couche] ; i++)
				for(int j = 0 ; j < NbNeurones[couche - 1] ; j ++)
					poids[couche - 1][i][j] += -1 * coeffApprent * deltas[couche-1][i] * S[couche-1][j];
		}
	}   
	private static void lectureFichier() {
		// lecture des donnees a partir du fichier iris.data
		String ligne, sousChaine;
		int classe=0, n=0;
		try {
			BufferedReader fic=new BufferedReader(new FileReader("Iris/iris.data"));
			while ((ligne=fic.readLine())!=null) {
				for(int i=0;i<NbCaract;i++) {
					if(ligne.length()>0)
					{
						sousChaine = ligne.substring(i*NbCaract, i*NbCaract+3);
						data[classe][n][i] = Double.parseDouble(sousChaine);
						//System.out.println(data[classe][n][i]+" "+classe+" "+n);
					}
				}
				if (++n==NbEx) { n=0; classe++; }
			}
			fic.close();
		}
		catch (Exception e) { System.out.println(e.toString()); }
	}
	

	// MAIN
	public static void main(String[] args) {
		System.out.println("Caches="+NbCaches+" App="+NbApprent+" coef="+coeffApprent);
		initialisation();
		apprentissage();
		evaluation();
	}   
}  //------------------fin classe MLP--------------------