import networkx as nx

def flatten(L):
	if L == []:
		return []
	elif isinstance(L[0], int):
		return [L[0]] + flatten(L[1:])
	else:
		return flatten(L[0]) + flatten(L[1:])

def getLastElementOnLevel(L):
	if L == []:
		return None
	elif isinstance(L[-1], list):
		return getLastElementOnLevel(L[:-1])
	else:
		return L[-1]



class Layer:
	def __init__(self):
		self.operators = {}
		self.operators["loop"] = False
		self.operators["optional"] = False
		self.operators["relDepth"] = 0
		self.operators["eitherOr"] = False
		self.elements = []
		self.parentLayer = None
		self.childLayers = []

	def printLayer(self):
		print "Layer Operators:"
		print("LOOP: ", self.operators["loop"])
		print("OPTIONAL:", self.operators["optional"])
		print("DEPTH:", self.operators["relDepth"])
		print("EITHER OR:", self.operators["eitherOr"])

		for element in self.elements:
			if isinstance(element, str):
				print("ELEMENT: ", element)
			else:
				element.printLayer()


HHMM = nx.DiGraph()

HHMM.add_node(0, {"name": "root", "root": True, "internal": True, "production": False, "productionValues": [], "parent": [], "children": [], "level": 0, "position": (100,100)})

ruleToNode = {}

grammarFile = open("JavaParser.g4", 'r')
#grammarFile = open("JavaParser1.g4", 'r')
#grammarFile = open("expression.g4", 'r')

# Getting Rule Names -> Internal States
lineCounter = 0
nodeCounter = 1
inRule = False
inComment = False
for line in grammarFile:
	if lineCounter > 32:
		lineList = line.split()
		if lineList != []:
			if lineList[0] == "/*":
				inComment = True
			elif lineList[0] == "*/":
				inComment = False
			elif inRule and lineList[0] == ';':
				inRule = False
			elif lineList[0] not in ["//", "/**", "*/", '*', '/*'] and not inRule and not inComment:
				if lineList[0] not in [';', '|', '(', ':', ')']:
					inRule = True
					HHMM.add_node(nodeCounter, {"name": lineList[0], "root": False, "internal": True, "production": False, "productionValues":[], "parent": [], "children": [], "level": None, "position": (0,0)})
					ruleToNode[lineList[0]] = nodeCounter
					nodeCounter += 1

	lineCounter += 1

# Elements in the Grammar File that only appear once should be at the top of the hierarchy
ruleCounts = {}
ruleEntries = {}
for key in ruleToNode.keys():
	ruleCounts[key] = 0
	ruleEntries[key] = {}
	ruleEntries[key]["entries"] = []

grammarFile = open("JavaParser.g4", 'r')
#grammarFile = open("JavaParser1.g4", 'r')
#grammarFile = open("expression.g4", 'r')

inRule = False
parenDepth = 0
currNode = 0
entryLayer = Layer()
currLayer = entryLayer
inInsertedParen = False
rule = ""
eitherOrCounter = 0
for line in grammarFile:
	if lineCounter > 32:
		lineList = line.split()
		if lineList != []:
			if lineList[0] in ruleToNode.keys() and not inRule:
				inRule = True
				currNode = ruleToNode[lineList[0]]
				rule = lineList[0]
				parenDepth = 0
				entryLayer = Layer()
				currLayer = entryLayer
				eitherOrCounter = 0
				#print "NEW RULE"
			elif lineList[0] == ';':
				inRule = False
				ruleEntries[rule]["entries"] += [entryLayer]
			elif lineList[0] in ['|', ':'] and inRule:
				if lineList[0] == '|' and parenDepth > 0:
					if eitherOrCounter == 0:
						leftLayer = Layer()
						rightLayer = Layer()
						eitherOrLayer = currLayer
						eitherOrLayer.operators["eitherOr"] = True
						leftLayer.operators["relDepth"] = eitherOrLayer.operators["relDepth"]
						rightLayer.operators["relDepth"] = eitherOrLayer.operators["relDepth"]
						leftLayer.elements = eitherOrLayer.elements
						eitherOrLayer.elements = [leftLayer]
						leftLayer.parentLayer = eitherOrLayer
						rightLayer.parentLayer = eitherOrLayer
						currLayer = rightLayer
						eitherOrCounter += 1
					else:
						eitherOrLayer = currLayer.parentLayer
						eitherOrLayer.elements += [currLayer]
						newRightLayer = Layer()
						newRightLayer.operators["relDepth"] = eitherOrLayer.operators["relDepth"]
						newRightLayer.parentLayer = eitherOrLayer
						currLayer = newRightLayer
						eitherOrCounter += 1
						
						"""
						#print "WITHIN ENTRY 1"
						currLayer.operators["eitherOr"] = True
						# Hack to have each line of (...|...|...) list as a layer
						lineList[1] = "(" + lineList[1]
						lineList[-1] = lineList[-1] + ")"
						inInsertedParen = True
						#print lineList
						"""
				elif lineList[0] == '|':
					#print "NEW ENTRY 1"
					ruleEntries[rule]["entries"] += [entryLayer]
					entryLayer = Layer()
					currLayer = entryLayer
					eitherOrCounter = 0
				else:
					eitherOrCounter = 0
					#print "NEW ENTRY 2"
				for element in lineList[1:]:
					if element == "//":
						break
					if element[0] == '<':
						continue
					if element[0] == '(':
						parenDepth += 1
						newLayer = Layer()
						newLayer.operators["relDepth"] = parenDepth
						currLayer.childLayers += [newLayer]
						newLayer.parentLayer = currLayer
						currLayer = newLayer
						element = element[1:]
						# There's instances of "(("
						if len(element) > 0 and element[0] == '(':
							newLayer = Layer()
							currLayer.childLayers += [newLayer]
							newLayer.parentLayer = currLayer
							currLayer = newLayer
							parenDepth += 1
							currLayer.operators["relDepth"] = parenDepth
							element = element[1:]
					elif element[0] == '|':
						element = element[1:]
						if eitherOrCounter == 0:
							leftLayer = Layer()
							rightLayer = Layer()
							eitherOrLayer = currLayer
							eitherOrLayer.operators["eitherOr"] = True
							leftLayer.operators["relDepth"] = parenDepth
							rightLayer.operators["relDepth"] = parenDepth
							leftLayer.elements = eitherOrLayer.elements
							eitherOrLayer.elements = [leftLayer]
							leftLayer.parentLayer = eitherOrLayer
							rightLayer.parentLayer = eitherOrLayer
							currLayer = rightLayer
							eitherOrCounter += 1
						else:
							eitherOrLayer = currLayer.parentLayer
							eitherOrLayer.elements += [currLayer]
							newRightLayer = Layer()
							newRightLayer.operators["relDepth"] = parenDepth
							newRightLayer.parentLayer = eitherOrLayer
							currLayer = newRightLayer
							eitherOrCounter += 1

					if "=" in element and "'='" not in element:
						equalPosition = element.find("=")
						element = element[equalPosition + 1:]
					

					if len(element) > 0 and element[0] == '(':
						newLayer = Layer()
						currLayer.childLayers += [newLayer]
						newLayer.parentLayer = currLayer
						currLayer = newLayer
						parenDepth += 1
						element = element[1:]
						# There's instances of "(("
						if len(element) > 0 and element[0] == '(':
							newLayer = Layer()
							currLayer.childLayers += [newLayer]
							newLayer.parentLayer = currLayer
							currLayer = newLayer
							parenDepth += 1
							element = element[1:]

					if len(element) != 0 and element[-1] == '?':
						if element[-4:-1] == ")*)":
							element = element[:-4]
							currLayer.operators["relDepth"] = parenDepth
							currLayer.operators["optional"] = True
							currLayer.operators["loop"] = True
							currLayer.elements += [element]
							parenDepth -= 1
							parentLayer = currLayer.parentLayer
							parentLayer.operators["relDepth"] = parenDepth
							parentLayer.operators["optional"] = True
							parentLayer.elements += [currLayer]
							grandparentLayer = parentLayer.parentLayer
							grandparentLayer.elements += [parentLayer]
							parenDepth -= 1
							currLayer = grandparentLayer
						elif element[-2] == ')':
							element = element[:-2]
							if len(element) != 0:
								currLayer.elements += [element]
							currLayer.operators["relDepth"] = parenDepth
							parentLayer = currLayer.parentLayer
							if parentLayer.operators["eitherOr"] == True:
								parentLayer.elements += [currLayer]
								parentLayer.operators["optional"] = True
								grandparentLayer = parentLayer.parentLayer
								grandparentLayer.elements += [parentLayer]
								currLayer = grandparentLayer
							else:
								currLayer.operators["optional"] = True
								parentLayer.elements += [currLayer]
								currLayer = parentLayer
							parenDepth -= 1
						else:
							element = element[:-1]
							tempLayer = Layer()
							tempLayer.operators["relDepth"] = parenDepth
							tempLayer.operators["optional"] = True
							if len(element) != 0:
								tempLayer.elements += [element]
							currLayer.elements += [tempLayer]
					elif len(element) != 0 and element[-1] == '*':
						if element[-3:-1] == "?)":
							element = element[:-3]
							tempLayer = Layer()
							tempLayer.operators["relDepth"] = parenDepth
							tempLayer.operators["optional"] = True
							if len(element) != 0:
								tempLayer.elements += [element]
							currLayer.elements += [tempLayer]
							currLayer.operators["relDepth"] = parenDepth
							currLayer.operators["loop"] = True
							currLayer.operators["optional"] = True
							parentLayer = currLayer.parentLayer
							parentLayer.elements += [currLayer]
							currLayer = parentLayer
							parenDepth -= 1
						elif element[-2] == ')':
							element = element[:-2]
							if len(element) != 0:
								currLayer.elements += [element]
							currLayer.operators["relDepth"] = parenDepth
							currLayer.operators["loop"] = True
							currLayer.operators["optional"] = True
							parentLayer = currLayer.parentLayer
							parentLayer.elements += [currLayer]
							currLayer = parentLayer
							parenDepth -= 1
						else:
							element = element[:-1]
							tempLayer = Layer()
							tempLayer.operators["relDepth"] = parenDepth
							tempLayer.operators["loop"] = True
							tempLayer.operators["optional"] = True
							if len(element) != 0:
								tempLayer.elements += [element]
							currLayer.elements += [tempLayer]
					elif len(element) != 0 and element[-1] == '+':
						if element[-2] == ')':
							element = element[:-2]
							if len(element) != 0:
								currLayer.elements += [element]
							currLayer.operators["relDepth"] = parenDepth
							currLayer.operators["loop"] = True
							currLayer.operators["optional"] = False
							parentLayer = currLayer.parentLayer
							parentLayer.elements += [currLayer]
							currLayer = parentLayer
							parenDepth -= 1
						else:
							element = element[:-1]
							tempLayer = Layer()
							tempLayer.operators["relDepth"] = parenDepth
							tempLayer.operators["loop"] = True
							tempLayer.operators["optional"] = False
							if len(element) != 0:
								tempLayer.elements += [element]
							currLayer.elements += [tempLayer]
					elif len(element) != 0 and element[-1] == ')':
						if element[-3:-1] == ")*":
							element = element[:-3]
							currLayer.elements += [element]
							currLayer.operators["relDepth"] = parenDepth
							currLayer.operators["loop"] = True
							currLayer.operators["optional"] = True
							parenDepth -= 1
							parentLayer = currLayer.parentLayer
							parentLayer.elements += [currLayer]
							parentLayer.operators["relDepth"] = parenDepth
							grandparentLayer = parentLayer.parentLayer
							grandparentLayer.elements += [parentLayer]
							if grandparentLayer.operators["eitherOr"] == True:
								greatgrandparentLayer = grandparentLayer.parentLayer
								greatgrandparentLayer.elements += [grandparentLayer]
								currLayer = greatgrandparentLayer
							else:
								currLayer = grandparentLayer
						else:
							# Has to be end of an "eitherOr"
							element = element[:-1]
							if len(element) != 0:
								currLayer.elements += [element]
							
							eitherOrLayer = currLayer.parentLayer
							eitherOrLayer.elements += [currLayer]

							grandparentLayer = eitherOrLayer.parentLayer
							grandparentLayer.elements += [eitherOrLayer]

							parenDepth -= 1

							currLayer = grandparentLayer

							"""
							# Case where "eitherOr" is over multiple lines
							# layer(layer | layer | layer | ...)
							if inInsertedParen == True:
								parenDepth -= 1
								parentLayer = currLayer.parentLayer
								parentLayer.elements += [currLayer]
							# Case where "eitherOr" is within one line
							else:
								parenDepth -= 1

							parentLayer = currLayer.parentLayer
							parentLayer.elements += [currLayer]

							grandparentLayer = parentLayer.parentLayer
							grandparentLayer.elements += [parentLayer]
							
							currLayer = grandparentLayer

							currLayer.printLayer()
							

							if not inInsertedParen:
								parenDepth -= 1
							"""
					elif len(element) != 0:
						currLayer.elements += [element]

					if element in ruleCounts.keys():
						ruleCounts[element] += 1
			elif inRule:
				#print "WITHIN ENTRY 2"
				for element in lineList:
					if element == "//":
						break
					if element[0] == '<':
						continue
					if element[0] == '(':
						newLayer = Layer()
						currLayer.childLayers += [newLayer]
						newLayer.parentLayer = currLayer
						currLayer = newLayer
						parenDepth += 1
						element = element[1:]
						# There's instances of "(("
						if len(element) > 0 and element[0] == '(':
							newLayer = Layer()
							currLayer.childLayers += [newLayer]
							newLayer.parentLayer = currLayer
							currLayer = newLayer
							parenDepth += 1
							element = element[1:]
					elif element[0] == '|':
						element = element[1:]
						if eitherOrCounter == 0:
							leftLayer = Layer()
							rightLayer = Layer()
							eitherOrLayer = currLayer
							eitherOrLayer.operators["eitherOr"] = True
							leftLayer.operators["relDepth"] = parenDepth
							rightLayer.operators["relDepth"] = parenDepth
							leftLayer.elements = eitherOrLayer.elements
							eitherOrLayer.elements = [leftLayer]
							leftLayer.parentLayer = eitherOrLayer
							rightLayer.parentLayer = eitherOrLayer
							currLayer = rightLayer
							eitherOrCounter += 1
						else:
							eitherOrLayer = currLayer.parentLayer
							eitherOrLayer.elements += [currLayer]
							newRightLayer = Layer()
							newRightLayer.operators["relDepth"] = parenDepth
							newRightLayer.parentLayer = eitherOrLayer
							currLayer = newRightLayer
							eitherOrCounter += 1

					# Handling cases like "bop=("
					if "=" in element and "'='" not in element and element[0] != "'" and element[-1] != "'":
						equalPosition = element.find("=")
						element = element[equalPosition + 1:]
					elif "=" in element and "'='" in element and element.count("=") > 1:
						equalPosition = element.find("=")
						element = element[equalPosition + 1:]

					if len(element) > 0 and element[0] == '(':
						newLayer = Layer()
						currLayer.childLayers += [newLayer]
						newLayer.parentLayer = currLayer
						currLayer = newLayer
						parenDepth += 1
						element = element[1:]
						# There's instances of "(("
						if len(element) > 0 and element[0] == '(':
							newLayer = Layer()
							currLayer.childLayers += [newLayer]
							newLayer.parentLayer = currLayer
							currLayer = newLayer
							parenDepth += 1
							element = element[1:]

					if len(element) != 0 and element[-1] == '?':
						if element[-2] == ')':
							element = element[:-2]
							if len(element) != 0:
								currLayer.elements += [element]
							currLayer.operators["relDepth"] = parenDepth
							currLayer.operators["optional"] = True
							parentLayer = currLayer.parentLayer
							parentLayer.elements += [currLayer]
							currLayer = parentLayer
							parenDepth -= 1
						else:
							element = element[:-1]
							tempLayer = Layer()
							tempLayer.operators["relDepth"] = parenDepth
							tempLayer.operators["optional"] = True
							if len(element) != 0:
								tempLayer.elements += [element]
							currLayer.elements += [tempLayer]
					elif len(element) != 0 and element[-1] == '*':
						if element[-2] == ')':
							element = element[:-2]
							if len(element) != 0:
								currLayer.elements += [element]
							currLayer.operators["relDepth"] = parenDepth
							currLayer.operators["loop"] = True
							currLayer.operators["optional"] = True
							parentLayer = currLayer.parentLayer
							parentLayer.elements += [currLayer]
							currLayer = parentLayer
							parenDepth -= 1
						else:
							element = element[:-1]
							tempLayer = Layer()
							tempLayer.operators["relDepth"] = parenDepth
							tempLayer.operators["loop"] = True
							tempLayer.operators["optional"] = True
							if len(element) != 0:
								tempLayer.elements += [element]
							currLayer.elements += [tempLayer]
					elif len(element) != 0 and element[-1] == '+':
						if element[-2] == ')':
							element = element[:-2]
							if len(element) != 0:
								currLayer.elements += [element]
							currLayer.operators["relDepth"] = parenDepth
							currLayer.operators["loop"] = True
							currLayer.operators["optional"] = False
							parentLayer = currLayer.parentLayer
							parentLayer.elements += [currLayer]
							currLayer = parentLayer
							parenDepth -= 1
						else:
							element = element[:-1]
							tempLayer = Layer()
							tempLayer.operators["relDepth"] = parenDepth
							tempLayer.operators["loop"] = True
							tempLayer.operators["optional"] = False
							if len(element) != 0:
								tempLayer.elements += [element]
							currLayer.elements += [tempLayer]
					elif len(element) != 0 and element[-1] == ')':
						# Has to be end of an "eitherOr"
						element = element[:-1]
						if len(element) != 0:
							currLayer.elements += [element]
						
						eitherOrLayer = currLayer.parentLayer
						eitherOrLayer.elements += [currLayer]

						grandparentLayer = eitherOrLayer.parentLayer
						grandparentLayer.elements += [eitherOrLayer]

						parenDepth -= 1

						currLayer = grandparentLayer
					elif len(element) != 0:
						currLayer.elements += [element]

					if element in ruleCounts.keys():
						ruleCounts[element] += 1


print "classDeclaration:"
entries = ruleEntries["classDeclaration"]["entries"]

entryCounter = 0
for entry in entries:
	print("Entry:", entryCounter)
	entryCounter += 1
	entry.printLayer()