import numpy
import sys
import os
import errno
import matplotlib.pyplot as plt

MAX_NODOS = 200


def generar_graficos_knn(ejec,output,alpha,mode):
	valor_k = []
	recall = []
	precision = []
	f1 = []
	acc = []
	for k in [3,25,45,65,86,105,125,155,175]:
		print("k={}".format(k))
		valor_k.append(k)
		comm = "./{} -debug -m {} -alpha {} -k {}".format(ejec,mode,alpha,k)
		results = os.popen(comm).read()
		results = [int(s.strip()) for s in results.splitlines()]
		for v in results:
			print("           ")
			print("{} ".format(v))
		tp = results[0]
		fp = results[1]
		tn = results[2] 
		fn = results[3]
		tot = tp + tn + fp + fn
		r = 100*tp / float(tp + fn)
		p = 100*tp / float(tp + fp)
		recall.append(r)
		precision.append(p)
		f = 2*p*r/float(p+r)
		f1.append(f)
		a = 100*float((tp+tn))/float(tot)
		acc.append(a)
		print("K:{} ACC:{} Pre:{} Rec:{} F1:{} ".format(k,a,p,r,f))
			


	# if os.path.exists("output.txt"): #optional check if file exists
 #  		with open(file_path, 'a') as file:
 #    file.write("\n") # could be any text, appended @ the end of file

	miny = 50
	maxy = 100	
	index = numpy.arange(len(valor_k))
	bar_width = 0.35
	plt.ylim([miny, maxy])
	plt.bar(index,recall,bar_width, label="recall")
	plt.bar(index+bar_width,precision,bar_width,label="Precision")
	plt.xticks(index + bar_width / 2, valor_k)
	#figname = "ranking_grafo_estrella_0{}.png".format(int(p*10)) 
	figname = "{}.png".format(output)
	plt.xlabel('Cantidad de vecinos')
	plt.ylabel('Porcentaje')
	plt.title('Precision y Recall de KNN')
	plt.legend(loc='best')
	plt.grid(True)
	plt.savefig(figname)
	plt.close()

	#Grafico accuarcy y F1
	index = numpy.arange(len(valor_k))
	bar_width = 0.35
	plt.ylim([miny, maxy])
	plt.bar(index,acc,bar_width, label="Accuarcy")
	plt.bar(index+bar_width,f1,bar_width,label="Media armonica (F1)")
	plt.xticks(index + bar_width / 2, valor_k)
	#figname = "ranking_grafo_estrella_0{}.png".format(int(p*10)) 
	figname = "{}_acc.png".format(output)
	plt.xlabel('Cantidad de vecinos')
	plt.ylabel('Porcentaje')
	plt.title('Accuarcy y media amornica de KNN')
	plt.legend(loc='best')
	plt.grid(True)
	plt.savefig(figname)
	plt.close()

def graph_from_data(valor_k,tp,tn,fp,fn,output):
	tot = []
	recall = []
	precision = []
	f1 = []
	acc = []
	for i in range(0,len(valor_k)):
 		tot.append(tp[i] + tn[i] + fp[i] + fn[i])
		r = 100*tp[i] / (tp[i] + fn[i])
		p = 100*tp[i] / (tp[i] + fp[i])
		recall.append(r)
		precision.append(p)
		if (p+r)==0:
			f1.append(0)
		else:
			f1.append(2*p*r/(p+r))
		acc.append(100*float((tp[i]+tn[i]))/float(tot[i]))
		
	miny = 0
	maxy = 100	
	index = numpy.arange(len(valor_k))
	bar_width = 0.35
	plt.ylim([miny, maxy])
	plt.bar(index,recall,bar_width, label="recall")
	plt.bar(index+bar_width,precision,bar_width,label="Precision")
	plt.xticks(index + bar_width / 2, valor_k,rotation=45)
	#figname = "ranking_grafo_estrella_0{}.png".format(int(p*10)) 
	figname = "{}.png".format(output)
	plt.xlabel('Cantidad de vecinos')
	plt.ylabel('Porcentaje')
	plt.title('Precision y Recall de KNN')
	plt.legend(loc='best')
	plt.grid(True)
	plt.savefig(figname)
	plt.close()

	#Grafico accuarcy y F1
	index = numpy.arange(len(valor_k))
	bar_width = 0.35
	plt.ylim([miny, maxy])
	plt.bar(index,acc,bar_width, label="Accuarcy")
	plt.bar(index+bar_width,f1,bar_width,label="Media armonica (F1)")
	plt.xticks(index + bar_width / 2, valor_k,rotation=45)
	#figname = "ranking_grafo_estrella_0{}.png".format(int(p*10)) 
	figname = "{}_acc.png".format(output)
	plt.xlabel('Cantidad de vecinos')
	plt.ylabel('Porcentaje')
	plt.title('Accuarcy y media amornica de KNN')
	plt.legend(loc='best')
	plt.grid(True)
	plt.savefig(figname)
	plt.close()



if __name__ == "__main__":
    
	ejec ="tp2b"
	output="out"


	if len(sys.argv)<5:
		print("Faltan argumentos: 'generador.py tp2 output alpha modo'")
		exit(0)
	else:
		ejec = str(sys.argv[1])
		output = str(sys.argv[2])
		alpha = int(sys.argv[3])
		mode = int(sys.argv[4])
	
	if os.path.isfile("{}".format(ejec)): 
		print("generando grafico con alpha {}".format(alpha))
		generar_graficos_knn(ejec,output,alpha,mode)
	else:
		print("No se encontro ejecutable ./{}").format(ejec)







